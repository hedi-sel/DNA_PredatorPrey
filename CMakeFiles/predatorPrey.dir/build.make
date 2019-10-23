# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/hedi/.cmake/bin/cmake

# The command to remove a file.
RM = /home/hedi/.cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hedi/Documents/DNA_PredatorPrey

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hedi/Documents/DNA_PredatorPrey

# Include any dependencies generated for this target.
include CMakeFiles/predatorPrey.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/predatorPrey.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/predatorPrey.dir/flags.make

CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.o: CMakeFiles/predatorPrey.dir/flags.make
CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.o: src/PdeSystem/cudaComputer.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hedi/Documents/DNA_PredatorPrey/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.o"
	nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/hedi/Documents/DNA_PredatorPrey/src/PdeSystem/cudaComputer.cu -o CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.o

CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.o: CMakeFiles/predatorPrey.dir/flags.make
CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.o: src/PdeSystem/predator_prey_system.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hedi/Documents/DNA_PredatorPrey/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.o -c /home/hedi/Documents/DNA_PredatorPrey/src/PdeSystem/predator_prey_system.cpp

CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hedi/Documents/DNA_PredatorPrey/src/PdeSystem/predator_prey_system.cpp > CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.i

CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hedi/Documents/DNA_PredatorPrey/src/PdeSystem/predator_prey_system.cpp -o CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.s

CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.o: CMakeFiles/predatorPrey.dir/flags.make
CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.o: src/PdeSystem/predator_prey_system_gpu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hedi/Documents/DNA_PredatorPrey/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.o -c /home/hedi/Documents/DNA_PredatorPrey/src/PdeSystem/predator_prey_system_gpu.cpp

CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hedi/Documents/DNA_PredatorPrey/src/PdeSystem/predator_prey_system_gpu.cpp > CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.i

CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hedi/Documents/DNA_PredatorPrey/src/PdeSystem/predator_prey_system_gpu.cpp -o CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.s

CMakeFiles/predatorPrey.dir/src/main.cpp.o: CMakeFiles/predatorPrey.dir/flags.make
CMakeFiles/predatorPrey.dir/src/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hedi/Documents/DNA_PredatorPrey/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/predatorPrey.dir/src/main.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/predatorPrey.dir/src/main.cpp.o -c /home/hedi/Documents/DNA_PredatorPrey/src/main.cpp

CMakeFiles/predatorPrey.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/predatorPrey.dir/src/main.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hedi/Documents/DNA_PredatorPrey/src/main.cpp > CMakeFiles/predatorPrey.dir/src/main.cpp.i

CMakeFiles/predatorPrey.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/predatorPrey.dir/src/main.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hedi/Documents/DNA_PredatorPrey/src/main.cpp -o CMakeFiles/predatorPrey.dir/src/main.cpp.s

# Object files for target predatorPrey
predatorPrey_OBJECTS = \
"CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.o" \
"CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.o" \
"CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.o" \
"CMakeFiles/predatorPrey.dir/src/main.cpp.o"

# External object files for target predatorPrey
predatorPrey_EXTERNAL_OBJECTS =

CMakeFiles/predatorPrey.dir/cmake_device_link.o: CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.o
CMakeFiles/predatorPrey.dir/cmake_device_link.o: CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.o
CMakeFiles/predatorPrey.dir/cmake_device_link.o: CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.o
CMakeFiles/predatorPrey.dir/cmake_device_link.o: CMakeFiles/predatorPrey.dir/src/main.cpp.o
CMakeFiles/predatorPrey.dir/cmake_device_link.o: CMakeFiles/predatorPrey.dir/build.make
CMakeFiles/predatorPrey.dir/cmake_device_link.o: CMakeFiles/predatorPrey.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hedi/Documents/DNA_PredatorPrey/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CUDA device code CMakeFiles/predatorPrey.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/predatorPrey.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/predatorPrey.dir/build: CMakeFiles/predatorPrey.dir/cmake_device_link.o

.PHONY : CMakeFiles/predatorPrey.dir/build

# Object files for target predatorPrey
predatorPrey_OBJECTS = \
"CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.o" \
"CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.o" \
"CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.o" \
"CMakeFiles/predatorPrey.dir/src/main.cpp.o"

# External object files for target predatorPrey
predatorPrey_EXTERNAL_OBJECTS =

run/predatorPrey: CMakeFiles/predatorPrey.dir/src/PdeSystem/cudaComputer.cu.o
run/predatorPrey: CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system.cpp.o
run/predatorPrey: CMakeFiles/predatorPrey.dir/src/PdeSystem/predator_prey_system_gpu.cpp.o
run/predatorPrey: CMakeFiles/predatorPrey.dir/src/main.cpp.o
run/predatorPrey: CMakeFiles/predatorPrey.dir/build.make
run/predatorPrey: CMakeFiles/predatorPrey.dir/cmake_device_link.o
run/predatorPrey: CMakeFiles/predatorPrey.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hedi/Documents/DNA_PredatorPrey/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable run/predatorPrey"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/predatorPrey.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/predatorPrey.dir/build: run/predatorPrey

.PHONY : CMakeFiles/predatorPrey.dir/build

CMakeFiles/predatorPrey.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/predatorPrey.dir/cmake_clean.cmake
.PHONY : CMakeFiles/predatorPrey.dir/clean

CMakeFiles/predatorPrey.dir/depend:
	cd /home/hedi/Documents/DNA_PredatorPrey && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hedi/Documents/DNA_PredatorPrey /home/hedi/Documents/DNA_PredatorPrey /home/hedi/Documents/DNA_PredatorPrey /home/hedi/Documents/DNA_PredatorPrey /home/hedi/Documents/DNA_PredatorPrey/CMakeFiles/predatorPrey.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/predatorPrey.dir/depend

