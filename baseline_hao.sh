for group_name in G1_instruction G1_category G1_tool G2_instruction G2_category
do
    bash hao_inference.sh $group_name CoT@1
done