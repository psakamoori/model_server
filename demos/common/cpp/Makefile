#
# Copyright (c) 2021 Intel Corporation
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
#

ovms_cpp_image_classification:
	@cp ../../../third_party/opencv/install_opencv.sh ../../../third_party/opencv/opencv_cmake_flags.txt  third_party/opencv
	@docker build -t ovms_cpp_image_classification:latest --build-arg CLIENT=ovms_cpp_image_classification --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} .
	@rm third_party/opencv/install_opencv.sh third_party/opencv/opencv_cmake_flags.txt

ovms_cpp_benchmark:
	@cp ../../../third_party/opencv/install_opencv.sh ../../../third_party/opencv/opencv_cmake_flags.txt  third_party/opencv
	@docker build -t ovms_cpp_benchmark:latest --build-arg CLIENT=ovms_cpp_benchmark --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} --build-arg no_proxy=${no_proxy} .
	@rm third_party/opencv/install_opencv.sh third_party/opencv/opencv_cmake_flags.txt