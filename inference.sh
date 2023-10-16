export TOOLBENCH_KEY=xud2dQdMNE8MRlkwBTe5DYB0DkktPi1uiCnyteptzmg3gS7Bbz
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./


export OPENAI_API_KEY=openchat 
OPENAI_KEY=$OPENAI_API_KEY
export OPENAI_API_BASE="https://api.01ww.xyz/v1" 
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model chatgpt_function \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method CoT@1 \
    --input_query_file data/instruction/G2_query.json \
    --output_answer_file data/answer/chatgpt_buffer_cot_g2_test_set_v3_record \
    --toolbench_key $TOOLBENCH_KEY