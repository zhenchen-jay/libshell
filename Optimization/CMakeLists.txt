add_library(optimization)
message(STATUS "creating target 'optimization'")

file(GLOB_RECURSE INC_FILES "include/*.h")
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
target_sources(optimization PRIVATE ${INC_FILES} ${SRC_FILES})

target_link_libraries(optimization
    PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
)

# Platform-specific configuration
if(APPLE)
    option(OPT_SOLVER_WITH_ACCELERATE "Accelerate support for linear solver" ON)
    if(OPT_SOLVER_WITH_ACCELERATE)
        find_package(BLAS REQUIRED)
        target_link_libraries(optimization PUBLIC ${BLAS_LIBRARIES})
        target_compile_definitions(optimization PUBLIC OPT_SOLVER_ACCELERATE)
    endif()
else()
    set(OPT_SOLVER_WITH_ACCELERATE OFF CACHE BOOL "Accelerate support for linear solver" FORCE)
endif()

# Add platform-specific definitions
if(APPLE)
    target_compile_definitions(optimization PUBLIC PLATFORM_APPLE)
elseif(WIN32)
    target_compile_definitions(optimization PUBLIC PLATFORM_WINDOWS)
elseif(UNIX)
    target_compile_definitions(optimization PUBLIC PLATFORM_LINUX)
endif()
