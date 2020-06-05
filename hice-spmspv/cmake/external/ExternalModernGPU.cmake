set(target_name moderngpu)

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/ExternalModernGPU.cmake.in
    ${EXTERNAL_PROJECT_DIR}/${target_name}/CMakeLists.txt
    @ONLY
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E make_directory super 
  WORKING_DIRECTORY ${EXTERNAL_PROJECT_DIR}/${target_name}
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .. 
  WORKING_DIRECTORY ${EXTERNAL_PROJECT_DIR}/${target_name}/super
)

execute_process(
  COMMAND ${CMAKE_COMMAND} --build . 
  WORKING_DIRECTORY ${EXTERNAL_PROJECT_DIR}/${target_name}/super
)

add_library(${target_name} INTERFACE)

target_include_directories(${target_name} 
  INTERFACE
    $<BUILD_INTERFACE:${EXTERNAL_PROJECT_DIR}/${target_name}/include>
    $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}/${target_name}>
)
