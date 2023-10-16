export TOOLBENCH_KEY="xud2dQdMNE8MRlkwBTe5DYB0DkktPi1uiCnyteptzmg3gS7Bbz"
export OUTPUT_DIR="data/answer/chatgpt_dfs_2"
export OPENAI_API_BASE="https://api.01ww.xyz/v1" 
export OPENAI_KEY="openchat"
export PYTHONPATH=./

mkdir $OUTPUT_DIR
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model chatgpt_function \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method DFS_woFilter_w2 \
    --input_query_file data/instruction/inference_query_demo.json \
    --output_answer_file $OUTPUT_DIR \
    --toolbench_key $TOOLBENCH_KEY

