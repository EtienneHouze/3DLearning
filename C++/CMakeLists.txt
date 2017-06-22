cmake_minimum_required(VERSION 3.0.0)
project(AltitudeRender)

SET("OpenCV_DIR" "${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/OpenCV/build")
find_package(OpenCV REQUIRED)

find_package(OpenGL REQUIRED)

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
	${OpenCV_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIRS}
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/boost/include/boost-1_63"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/CGAL/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/GLEW/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/FreeGLUT/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/GLM"
	)
	
link_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/boost/lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/CGAL/lib"
	)
	
file(GLOB SOURCES src/*.cpp src/*.h)

add_executable(AltitudeRender AltitudeRender.cpp ${SOURCES})
target_link_libraries(AltitudeRender
	${OpenCV_LIBS} 
	${CGAL_LIBRARIES} 
	${OPENGL_LIBRARIES} 
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/GLEW/lib/Release/x64/glew32.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/FreeGLUT/lib/x64/freeglut.lib"
	)

add_executable(DepthRender DepthRender.cpp ${SOURCES})
target_link_libraries(DepthRender
	${OpenCV_LIBS} 
	${CGAL_LIBRARIES} 
	${OPENGL_LIBRARIES} 
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/GLEW/lib/Release/x64/glew32.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/FreeGLUT/lib/x64/freeglut.lib"
	)

add_executable(PointProjection PointProjection.cpp ${SOURCES})
target_link_libraries(PointProjection
	${OpenCV_LIBS} 
	${CGAL_LIBRARIES} 
	${OPENGL_LIBRARIES} 
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/GLEW/lib/Release/x64/glew32.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/FreeGLUT/lib/x64/freeglut.lib"
	)

add_executable(TextureRender TextureRender.cpp ${SOURCES})
target_link_libraries(TextureRender
	${OpenCV_LIBS} 
	${CGAL_LIBRARIES} 
	${OPENGL_LIBRARIES} 
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/GLEW/lib/Release/x64/glew32.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/FreeGLUT/lib/x64/freeglut.lib"
	)

add_executable(LabelRender LabelRender.cpp ${SOURCES})
target_link_libraries(LabelRender
	${OpenCV_LIBS} 
	${CGAL_LIBRARIES} 
	${OPENGL_LIBRARIES} 
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/GLEW/lib/Release/x64/glew32.lib"
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/FreeGLUT/lib/x64/freeglut.lib"
	)

 SET(DLLs 
	"${OpenCV_DIR}/x64/vc14/bin/opencv_world320.dll" 
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/FreeGLUT/bin/x64/freeglut.dll" 
	"${CMAKE_CURRENT_SOURCE_DIR}/3rd_Party/GLEW/bin/Release/x64/glew32.dll" 
	)

foreach(dll ${DLLs})
	 add_custom_command(TARGET AltitudeRender POST_BUILD
		 COMMAND ${CMAKE_COMMAND} -E copy 
			 "${dll}"
			 "${CMAKE_CURRENT_BINARY_DIR}/Release")
	 add_custom_command(TARGET PointProjection POST_BUILD
		 COMMAND ${CMAKE_COMMAND} -E copy 
			 "${dll}"
			 "${CMAKE_CURRENT_BINARY_DIR}/Release")
endforeach() 

