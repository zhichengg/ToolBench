# export TOOLBENCH_KEY=xud2dQdMNE8MRlkwBTe5DYB0DkktPi1uiCnyteptzmg3gS7Bbz
# export RAPID_API_KEY=34e6e91265msh0157e1fc6794904p1f8a87jsn381998a1c4ee
# export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH=./


# export OPENAI_API_KEY=openchat 
# OPENAI_KEY=$OPENAI_API_KEY
# export OPENAI_API_BASE="https://api.01ww.xyz/v1" 
# /ML-A100/home/csj/miniconda3/envs/open/bin/python toolbench/inference/qa_pipeline.py \
#     --tool_root_dir data/toolenv/tools \
#     --backbone_model chatgpt_function \
#     --openai_key $OPENAI_KEY \
#     --max_observation_length 1024 \
#     --observ_compress_method truncate \
#     --method CoT@1 \
#     --input_query_file data/test_instruction/G1_instruction.json \
#     --output_answer_file data/answer/hao/sidecar_raw/G1_instruction \
#     --rapidapi_key $RAPID_API_KEY \
#     --use_rapidapi_key 

#     # --toolbench_key $TOOLBENCH_KEY


# export TOOLBENCH_KEY=xud2dQdMNE8MRlkwBTe5DYB0DkktPi1uiCnyteptzmg3gS7Bbz
# export RAPID_API_KEY=34e6e91265msh0157e1fc6794904p1f8a87jsn381998a1c4ee
# export CUDA_VISIBLE_DEVICES=0



# export OPENAI_API_KEY=openchat 
# OPENAI_KEY=$OPENAI_API_KEY
# export OPENAI_API_BASE="https://api.01ww.xyz/v1" 

# # group=$1
# backbone_model=chatgpt_function
# method=CoT@1

# ROOT=/ML-A100/Home/csj/gzc/STC
# TB_ROOT=$ROOT/ToolBench

# cd $TB_ROOT
# export PYTHONPATH=./
# export NO_PROXY="8.218.239.54"

# BUFFER_TYPE=mock
# for group_name in G1_instruction G1_category G1_tool G2_instruction G2_category
# do
#     /ML-A100/home/csj/miniconda3/envs/open/bin/python toolbench/inference/qa_pipeline.py \
#     --tool_root_dir data/toolenv/tools \
#     --backbone_model $backbone_model \
#     --openai_key $OPENAI_KEY \
#     --max_observation_length 1024 \
#     --observ_compress_method truncate \
#     --method $method \
#     --input_query_file $TB_ROOT/data/test_instruction/$group_name.json \
#     --output_answer_file $ROOT/data/answer/hao/sidecar_${BUFFER_TYPE}/${backbone_model}_${method}_${group_name}_record \
#     --history_buffer ${BUFFER_TYPE} \
#     --toolbench_key $TOOLBENCH_KEY
# done

# BUFFER_TYPE=raw
# for group_name in G1_instruction G1_category G1_tool G2_instruction G2_category
# do
#     /ML-A100/home/csj/miniconda3/envs/open/bin/python toolbench/inference/qa_pipeline.py \
#     --tool_root_dir data/toolenv/tools \
#     --backbone_model $backbone_model \
#     --openai_key $OPENAI_KEY \
#     --max_observation_length 1024 \
#     --observ_compress_method truncate \
#     --method $method \
#     --input_query_file $TB_ROOT/data/test_instruction/$group_name.json \
#     --output_answer_file $ROOT/data/answer/hao/sidecar_${BUFFER_TYPE}/${backbone_model}_${method}_${group_name}_record \
#     --history_buffer ${BUFFER_TYPE} \
#     --toolbench_key $TOOLBENCH_KEY
# done

# BUFFER_TYPE=tool_desc
# for group_name in G1_instruction G1_category G1_tool G2_instruction G2_category
# do
#     /ML-A100/home/csj/miniconda3/envs/open/bin/python toolbench/inference/qa_pipeline.py \
#     --tool_root_dir data/toolenv/tools \
#     --backbone_model $backbone_model \
#     --openai_key $OPENAI_KEY \
#     --max_observation_length 1024 \
#     --observ_compress_method truncate \
#     --method $method \
#     --input_query_file $TB_ROOT/data/test_instruction/$group_name.json \
#     --output_answer_file $ROOT/data/answer/hao/sidecar_${BUFFER_TYPE}/${backbone_model}_${method}_${group_name}_record \
#     --history_buffer ${BUFFER_TYPE} \
#     --toolbench_key $TOOLBENCH_KEY
# done

# BUFFER_TYPE=raw
# for group_name in G1_instruction G1_category G1_tool G2_instruction G2_category
# do
#     /ML-A100/home/csj/miniconda3/envs/open/bin/python toolbench/inference/qa_pipeline.py \
#     --tool_root_dir data/toolenv/tools \
#     --backbone_model $backbone_model \
#     --openai_key $OPENAI_KEY \
#     --max_observation_length 1024 \
#     --observ_compress_method truncate \
#     --method DFS_woFilter_w2 \
#     --input_query_file $TB_ROOT/data/test_instruction/$group_name.json \
#     --output_answer_file $ROOT/data/answer/hao/dfs_sidecar_${BUFFER_TYPE}/${backbone_model}_${method}_${group_name}_record \
#     --history_buffer ${BUFFER_TYPE} \
#     --toolbench_key $TOOLBENCH_KEY
# done




export TOOLBENCH_KEY=xud2dQdMNE8MRlkwBTe5DYB0DkktPi1uiCnyteptzmg3gS7Bbz
export RAPID_API_KEY=34e6e91265msh0157e1fc6794904p1f8a87jsn381998a1c4ee
export CUDA_VISIBLE_DEVICES=0



export OPENAI_API_KEY=openchat 
OPENAI_KEY=$OPENAI_API_KEY
export OPENAI_API_BASE="https://api.01ww.xyz/v1" 

# group=$1
backbone_model=chatgpt_function
method=CoT@1
ROOT=/ML-A100/Home/csj/gzc/STC
TB_ROOT=$ROOT/ToolBench

cd $TB_ROOT
export PYTHONPATH=./
export NO_PROXY="8.218.239.54"


LOCAL_BUFFER=thought
for group_name in G1_instruction G1_category G1_tool G2_instruction G2_category
do
    /ML-A100/home/csj/miniconda3/envs/open/bin/python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools \
    --backbone_model $backbone_model \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method $method \
    --input_query_file $TB_ROOT/data/test_instruction/$group_name.json \
    --output_answer_file $ROOT/data/answer/hao/sidecar_local_${LOCAL_BUFFER}/${backbone_model}_${method}_${group_name}_record \
    --local_buffer ${LOCAL_BUFFER} \
    --toolbench_key $TOOLBENCH_KEY
done