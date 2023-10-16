export PYTHONPATH=./
export OPENAI_API_KEY=openchat 
OPENAI_KEY=$OPENAI_API_KEY
export OPENAI_API_BASE="https://api.01ww.xyz/v1" 



python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/ToolQAEnv \
    --backbone_model chatgpt_function \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method CoT@1 \
    --input_query_file test_customized.json \
    --output_answer_file data/answer/test_customized.output5 \
    --api_customization