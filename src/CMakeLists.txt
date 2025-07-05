# inference
set(Torch_DIR "/home/ogura-sota/genesis_ws/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
choreonoid_add_simple_controller(Go2InferenceController Go2InferenceController.cpp)
target_link_libraries(Go2InferenceController ${roscpp_LIBRARIES})
target_link_libraries(Go2InferenceController ${CHOREONOID_BASE_LIBRARIES} "${TORCH_LIBRARIES}")
set_property(TARGET Go2InferenceController PROPERTY CXX_STANDARD 17)
