#!/bin/bash

ROOT_DIR="Test"

# Loop through each genre folder
for genre_path in "$ROOT_DIR"/*; do
  if [ -d "$genre_path" ]; then
    genre=$(basename "$genre_path")
    index=0

    for file in "$genre_path"/*; do
      if [[ "$file" == *.mp3 || "$file" == *.wav ]]; then
        output="$genre_path/${genre}.${index}.wav"
        ffmpeg -y -ss 30 -t 30 -i "$file" "$output"
        ((index++))
      fi
    done
  fi
done

#
#GENRE_DIR="Test/blues"
#GENRE_NAME="classical"
#INDEX=0
#
#for file in "$GENRE_DIR"/*; do
#  if [[ "$file" == *.mp3 || "$file" == *.wav ]]; then
#    output="$GENRE_DIR/${GENRE_NAME}.${INDEX}.wav"
#    ffmpeg -y -ss 30 -t 30 -i "$file" "$output"
#    ((INDEX++))
#  fi
#done