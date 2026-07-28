#!/usr/bin/env bash
# Renderiza clips de demostración de J03WR9.
# Requiere venv activado y el checkpoint reports/cv/fold_J03WR9.pt.
#
# Uso:
#   ./render_all_clips.sh
#   THEME=light ./render_all_clips.sh
set -u  # Sin set -e para que la falla de un clip no detenga los demás

MATCH="J03WR9"
THEME="${THEME:-dark}"
FPS_VIDEO=30
OUTDIR="reports/video/demo"
mkdir -p "$OUTDIR"

# Formato: nombre|start_s|clip_seconds|slow|mode
CLIPS=(
  "facil|2216|12|2.0|full"
  "posesion_larga|908|24|1.5|full"
  "remate|2529|12|2.0|full"
  "corner|407|15|2.0|full"
  "showcase_oclusion|1750|14|2.0|full"
)

echo "=============================================================="
echo " Render de ${#CLIPS[@]} clips  ·  match ${MATCH}  ·  tema ${THEME}"
echo "=============================================================="
total_frames=0
for row in "${CLIPS[@]}"; do
  IFS="|" read -r name start clip slow mode <<< "$row"
  # Total frames por clip: hold = round(fps / 5 * slow)
  hold=$(python3 -c "print(max(1,round($FPS_VIDEO/5*$slow)))")
  nvid=$(python3 -c "print(int($clip*5*$hold))")
  total_frames=$((total_frames + nvid))
  printf "  %-18s start=%-5s clip=%-3ss slow=%-3s -> %5d frames de video\n" \
         "$name" "$start" "$clip" "$slow" "$nvid"
done
echo "  ----------------------------------------------------------"
printf "  TOTAL a renderizar: %d frames de video\n" "$total_frames"
echo "  (a esto se suma, por clip, ~20-40 s de carga + inferencia del modelo)"
echo "=============================================================="
echo ""

run_start=$SECONDS
ok=(); fail=()
i=0; N=${#CLIPS[@]}
for row in "${CLIPS[@]}"; do
  i=$((i + 1))
  IFS="|" read -r name start clip slow mode <<< "$row"
  out="${OUTDIR}/${name}_${MATCH}_${THEME}.mp4"
  echo ">>> [$i/$N] [$name]  start-s=$start  clip=${clip}s  slow=${slow}x  mode=$mode"
  t0=$SECONDS
  if python scripts/10_animate_ghosting.py \
        --match "$MATCH" --mode "$mode" \
        --start-s "$start" --clip-seconds "$clip" \
        --slow "$slow" --fps-video "$FPS_VIDEO" --trail-seconds 1.0 \
        --theme "$THEME" --device cpu \
        --out "$out"
  then
    dt=$((SECONDS - t0))
    elapsed=$((SECONDS - run_start))
    printf "    OK  %s   (%dm %02ds)   | acumulado %dm %02ds\n\n" \
           "$out" $((dt/60)) $((dt%60)) $((elapsed/60)) $((elapsed%60))
    ok+=("$name (${dt}s)")
  else
    dt=$((SECONDS - t0))
    printf "    FALLO tras %ds  (revisa el traceback de arriba)\n\n" "$dt"
    fail+=("$name")
  fi
done

total=$((SECONDS - run_start))
echo "=============================================================="
echo " Resumen"
echo "=============================================================="
printf "  Tiempo total: %dm %02ds\n" $((total/60)) $((total%60))
echo "  Clips OK (${#ok[@]}):"
for c in "${ok[@]}"; do echo "     - $c"; done
if [ "${#fail[@]}" -gt 0 ]; then
  echo "  Clips con fallo (${#fail[@]}):"
  for c in "${fail[@]}"; do echo "     - $c"; done
fi
echo "  Salidas en: ${OUTDIR}/"
echo "=============================================================="
