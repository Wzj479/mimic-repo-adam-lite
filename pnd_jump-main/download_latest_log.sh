#!/bin/bash

# 远程服务器配置
SERVER_USER="peter"
SERVER_IP="123.57.14.125"
SERVER_PORT="11022"

# 机器人类型（可通过参数指定）
ROBOT_TYPE="${1:-adam_lite_12dof}"  # 默认为 adam_lite_12dof

# 当前工作目录
CURRENT_DIR=$(pwd)

# 远程路径（与本地相同）
REMOTE_PATH="${CURRENT_DIR}"

# 本地logs目录
LOCAL_LOGS_DIR="${CURRENT_DIR}/logs/${ROBOT_TYPE}"

echo "=========================================="
echo "从远程服务器下载最新log"
echo "=========================================="
echo "服务器: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
echo "机器人类型: ${ROBOT_TYPE}"
echo "远程路径: ${REMOTE_PATH}"
echo "本地路径: ${LOCAL_LOGS_DIR}"
echo "=========================================="

# 检查本地logs目录是否存在
if [ ! -d "${LOCAL_LOGS_DIR}" ]; then
    echo "本地logs目录不存在，创建中..."
    mkdir -p "${LOCAL_LOGS_DIR}"
fi

# 获取远程服务器上最新的log目录
echo "正在查找远程服务器上的最新log..."
LATEST_LOG=$(ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} \
    "cd ${REMOTE_PATH}/logs/${ROBOT_TYPE} && ls -td */ 2>/dev/null | head -1 | sed 's/\///'" 2>&1)

# 检查SSH命令是否成功
if [ $? -ne 0 ]; then
    echo "错误: 无法连接到远程服务器或找不到logs目录"
    echo "${LATEST_LOG}"
    exit 1
fi

# 检查是否找到log目录
if [ -z "${LATEST_LOG}" ]; then
    echo "错误: 远程服务器上没有找到log目录"
    exit 1
fi

echo "找到最新的log目录: ${LATEST_LOG}"

# 询问用户是否继续
read -p "是否下载这个目录? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消下载"
    exit 0
fi

# 使用rsync下载最新的log目录
echo "开始下载..."
rsync -avz --progress \
    -e "ssh -p ${SERVER_PORT}" \
    ${SERVER_USER}@${SERVER_IP}:${REMOTE_PATH}/logs/${ROBOT_TYPE}/${LATEST_LOG} \
    ${LOCAL_LOGS_DIR}/

# 检查下载是否成功
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "下载成功!"
    echo "位置: ${LOCAL_LOGS_DIR}/${LATEST_LOG}"
    echo "=========================================="
    
    # 显示下载的目录大小
    du -sh "${LOCAL_LOGS_DIR}/${LATEST_LOG}"
else
    echo "错误: 下载失败"
    exit 1
fi

