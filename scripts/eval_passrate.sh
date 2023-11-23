# export CONVERTED_ANSWER_PATH=/ML-A100/Home/csj/gzc/STC/ToolBench/data/reproduction_data/model_predictions_converted/
export CONVERTED_ANSWER_PATH=/ML-A100/Home/csj/gzc/STC/data/converted_answer2
# export SAVE_PATH=data/pass_rate_results
export SAVE_PATH=/ML-A100/Home/csj/gzc/STC/data/pass_rate_results2
export CANDIDATE_MODEL=chatgpt_function_CoT@1
export API_POOL_FILE=/ML-A100/Home/csj/gzc/STC/openai_key.json
export OPENAI_API_BASE="https://api.01ww.xyz/v1" 

if [ ! -d ${SAVE_PATH} ]; then
    mkdir -p ${SAVE_PATH}
fi

python /ML-A100/Home/csj/gzc/STC/ToolBench/toolbench/tooleval/eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids /ML-A100/Home/csj/gzc/STC/ToolBench/data/test_query_ids \
    --max_eval_threads 100 \
    --evaluate_times 4
