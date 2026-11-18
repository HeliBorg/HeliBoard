# Copyright (C) 2011 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LOCAL_PATH := $(call my-dir)

############ some local flags
# If you change any of those flags, you need to rebuild both libjni_latinime_common_static
# and the shared library that uses libjni_latinime_common_static.
FLAG_DBG ?= false
FLAG_DO_PROFILE ?= false

######################################
include $(CLEAR_VARS)

LATIN_IME_SRC_DIR := src

LOCAL_C_INCLUDES += $(LOCAL_PATH)/$(LATIN_IME_SRC_DIR)

LOCAL_CFLAGS += -Wall -Wextra -Weffc++ -Wformat=2 -Wcast-qual -Wcast-align \
    -Wwrite-strings -Wfloat-equal -Wpointer-arith -Winit-self -Wredundant-decls \
    -Woverloaded-virtual -Wsign-promo -Wno-system-headers

# To suppress compiler warnings for unused variables/functions used for debug features etc.
LOCAL_CFLAGS += -Wno-unused-parameter -Wno-unused-function -w

# HACK: -mstackrealign is required for x86 builds running on pre-KitKat devices to avoid crashes
# with SSE instructions.
ifeq ($(TARGET_ARCH), x86)
    LOCAL_CFLAGS += -mstackrealign
endif # x86

include $(LOCAL_PATH)/NativeFileList.mk

LOCAL_SRC_FILES := \
    $(LATIN_IME_JNI_SRC_FILES) \
    $(addprefix $(LATIN_IME_SRC_DIR)/, $(LATIN_IME_CORE_SRC_FILES))

ifeq ($(FLAG_DO_PROFILE), true)
    $(warning Making profiling version of native library)
    LOCAL_CFLAGS += -DFLAG_DO_PROFILE -funwind-tables
else # FLAG_DO_PROFILE
ifeq ($(FLAG_DBG), true)
    $(warning Making debug version of native library)
    LOCAL_CFLAGS += -DFLAG_DBG -funwind-tables -fno-inline
ifeq ($(FLAG_FULL_DBG), true)
    $(warning Making full debug version of native library)
    LOCAL_CFLAGS += -DFLAG_FULL_DBG
endif # FLAG_FULL_DBG
endif # FLAG_DBG
endif # FLAG_DO_PROFILE

LOCAL_MODULE := libjni_latinime_common_static
LOCAL_MODULE_TAGS := optional

LOCAL_CLANG := true
LOCAL_SDK_VERSION := 14
LOCAL_NDK_STL_VARIANT := c++_static

include $(BUILD_STATIC_LIBRARY)
######################################
include $(CLEAR_VARS)

# All code in LOCAL_WHOLE_STATIC_LIBRARIES will be built into this shared library.
LOCAL_WHOLE_STATIC_LIBRARIES := libjni_latinime_common_static

ifeq ($(FLAG_DO_PROFILE), true)
    $(warning Making profiling version of native library)
    LOCAL_LDFLAGS += -llog
else # FLAG_DO_PROFILE
ifeq ($(FLAG_DBG), true)
    $(warning Making debug version of native library)
    LOCAL_LDFLAGS += -llog
endif # FLAG_DBG
endif # FLAG_DO_PROFILE

LOCAL_MODULE := libjni_latinime
LOCAL_MODULE_TAGS := optional

LOCAL_CLANG := true
LOCAL_SDK_VERSION := 14
LOCAL_NDK_STL_VARIANT := c++_static
LOCAL_LDFLAGS += -ldl

# Avoid issues with reproducible builds, see https://gitlab.com/fdroid/rfp/-/issues/2662
LOCAL_LDFLAGS += -Wl,--build-id=none

include $(BUILD_SHARED_LIBRARY)
#################### Clean up the tmp vars
include $(LOCAL_PATH)/CleanupNativeFileList.mk

#################### Unit test on host environment
#include $(LOCAL_PATH)/HostUnitTests.mk

#################### Unit test on target environment
#include $(LOCAL_PATH)/TargetUnitTests.mk
