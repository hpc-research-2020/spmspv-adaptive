set(target_name csr5)

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/ExternalCSR5.cmake.in
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
