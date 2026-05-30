import os
import sys
import yaml
import csv
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def filter_dataset(csv_path, selected_classes, output_csv_path):
    print(f"Caricamento del dataset completo da {csv_path}...")
    selected_set = set(selected_classes)
    filtered_rows = []
    
    if not os.path.exists(csv_path):
        print(f"Errore: il file {csv_path} non esiste nella directory corrente.")
        sys.exit(1)
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4:
                label = row[2].strip()
                if label in selected_set:
                    filtered_rows.append(row)
                    
    # Creazione directory di output se non esiste
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    print(f"Scrittura del subset filtrato in {output_csv_path}...")
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["youtube_id", "start_seconds", "label", "split"]) # Header
        writer.writerows(filtered_rows)
        
    print(f"Subset CSV salvato in: {output_csv_path} ({len(filtered_rows)} clip totali)")
    return [{"youtube_id": r[0], "start_seconds": r[1], "label": r[2], "split": r[3]} for r in filtered_rows]

def download_and_process_clip(row, audio_dir, video_frame_dir):
    yt_id = row["youtube_id"]
    start_sec = int(row["start_seconds"])
    label = row["label"]
    
    # Format per i nomi dei file
    base_name = f"{yt_id}_{start_sec}"
    audio_path = os.path.join(audio_dir, f"{base_name}.wav")
    frame_path = os.path.join(video_frame_dir, f"{base_name}.jpg")
    
    # Controllo se entrambi i file sono già stati scaricati ed elaborati (resume capability)
    if os.path.exists(audio_path) and os.path.exists(frame_path):
        return True, "Already downloaded"
        
    url = f"https://www.youtube.com/watch?v={yt_id}"
    
    try:
        # 1. Recupero degli URL dei flussi diretti tramite yt-dlp
        # Utilizziamo -g per stampare l'URL diretto senza scaricare il file intero
        cmd_ytdl = [
            "yt-dlp",
            "-g",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            url
        ]
        
        result = subprocess.run(cmd_ytdl, capture_output=True, text=True, check=True)
        urls = result.stdout.strip().split('\n')
        
        if len(urls) < 1:
            return False, "Failed to retrieve stream URLs"
            
        # Assegnazione flussi video e audio
        video_url = urls[0]
        audio_url = urls[1] if len(urls) > 1 else urls[0]
        
        # 2. Estrazione audio a 16kHz mono (durata 10s)
        if not os.path.exists(audio_path):
            cmd_ffmpeg_audio = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-t", "10",
                "-i", audio_url,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_path
            ]
            subprocess.run(cmd_ffmpeg_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
        # 3. Estrazione frame video centrale (secondo 5 della clip, cioè start_sec + 5)
        if not os.path.exists(frame_path):
            frame_sec = start_sec + 5
            cmd_ffmpeg_video = [
                "ffmpeg", "-y",
                "-ss", str(frame_sec),
                "-i", video_url,
                "-vframes", "1",
                "-q:v", "2",  # Alta qualità per JPEG
                "-vf", "scale=256:256",  # Pre-ridimensionamento leggero
                frame_path
            ]
            subprocess.run(cmd_ffmpeg_video, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
        return True, "Success"
        
    except subprocess.CalledProcessError as e:
        # Pulisce i file parziali se creati
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(frame_path):
            os.remove(frame_path)
        return False, f"Subprocess error: {str(e)}"
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(frame_path):
            os.remove(frame_path)
        return False, f"Unexpected error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Download e Preprocessing del subset VGGSound")
    parser.add_argument("--csv", type=str, default="vggsound.csv", help="Percorso al file vggsound.csv completo")
    parser.add_argument("--config", type=str, default="experiments/configs/common.yaml", help="Percorso al config common.yaml")
    parser.add_argument("--workers", type=str, default="4", help="Numero di thread concorrenti per il download")
    args = parser.parse_args()
    
    config = load_config(args.config)
    selected_classes = config["dataset"]["selected_classes"]
    root_dir = config["dataset"]["root"]
    
    # Directory per i dati
    subset_csv_path = os.path.join(root_dir, "subset.csv")
    audio_dir = os.path.join(root_dir, "audio")
    video_frame_dir = os.path.join(root_dir, "video_frames")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(video_frame_dir, exist_ok=True)
    
    # 1. Filtriamo il dataset se il file subset.csv non esiste ancora
    if not os.path.exists(subset_csv_path):
        rows = filter_dataset(args.csv, selected_classes, subset_csv_path)
    else:
        print(f"Subset CSV esistente trovato in: {subset_csv_path}")
        rows = []
        with open(subset_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                
    print(f"Inizio download delle clip ({len(rows)} totali)...")
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    workers = int(args.workers)
    print(f"Utilizzo di {workers} worker concorrenti per il download.")
    
    # Esecuzione con o senza tqdm
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_and_process_clip, row, audio_dir, video_frame_dir) 
            for row in rows
        ]
        for future in tqdm(futures, desc="Downloading", total=len(futures)):
            success, message = future.result()
            if success:
                if message == "Already downloaded":
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                failed_count += 1
                
    print("\n--- Download Completato ---")
    print(f"Scaricati con successo: {success_count}")
    print(f"Già presenti (saltati): {skipped_count}")
    print(f"Falliti (non disponibili): {failed_count}")

if __name__ == "__main__":
    main()
