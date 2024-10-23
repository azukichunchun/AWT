DATA=$1
TRAINER=awt_mixup

for SHOT in 1 2 4
do
    bash scripts/${TRAINER}/main.sh ${DATA} vit_b16_1_2_4_shot ${SHOT}
done

for SHOT in 8 16
do
    bash scripts/${TRAINER}/main.sh ${DATA} vit_b16_8_16_shot ${SHOT}
done
