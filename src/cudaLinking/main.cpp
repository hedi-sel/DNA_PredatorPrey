#include "play.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char ** argv)
{
	printf("starting");
	int n = 1000000;
	if(argc > 1)	{ n = atoi(argv[1]);}     // Number of particles
	if(argc > 2)	{	srand(atoi(argv[2])); } // Random seed
	play(n);
}