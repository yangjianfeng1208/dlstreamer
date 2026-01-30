#!/bin/bash
# ==============================================================================
# Copyright (C) 2020-2025 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

set -e
MODEL_NAME=${1:-all}

if [ -z "$MODELS_PATH" ]; then
  echo "MODELS_PATH is not specified"
  echo "Please set MODELS_PATH env variable with target path to download models"
  exit 1
fi

PYTHON_CMD=python3
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  PYTHON_CMD=python
fi

echo Downloading models to folder "$MODELS_PATH"

if ! "$PYTHON_CMD" -m pip show -qq openvino-dev || ! "$PYTHON_CMD" -m pip show -qq tensorflow; then
  echo "This script requires the Open Model Zoo Python modules and TensorFlow."
  echo "Please install them using the following command:"
  echo "${PYTHON_CMD} -m pip install tensorflow openvino-dev[onnx]"
  exit 1
fi

if [ "${MODEL_NAME}" == "all" ]; then
  mkdir -p "${MODELS_PATH}" && \
  omz_downloader --list "$(dirname "$0")"/models_omz_samples.lst -o "$MODELS_PATH" && \
  omz_converter --list "$(dirname "$0")"/models_omz_samples.lst -o "$MODELS_PATH" -d "$MODELS_PATH"
else
  echo "Downloading specified model: $MODEL_NAME"
  omz_downloader --name "$MODEL_NAME" -o "$MODELS_PATH" && \
  omz_converter --name "$MODEL_NAME" -o "$MODELS_PATH" -d "$MODELS_PATH"
fi
