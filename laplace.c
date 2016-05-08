#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*A program to solve 2D laplace equation in a rectangle using Jacobi convergence method*/
/*The program uses rowwise domain decomposition to achieve parallel computing*/
/* The rectangle is not limited to a square*/
/* The code does not require any modification to run on differnt number of processors*/


/* Author - Hrachya Hakobyan*/


/*------------INPUT PARAMETERS--------------*/

/*Dimensions of the matrix A exluding the boundaries
  Therefore, the actual size of the matrix is (n+2)x(m+2)*/
#define n  500
#define m  500
/*Number of processes used, must be evenly divisible by the number of rows*/
#define p 2
/*Rows per process*/
#define rpp n / p
/*Maximum iterations*/
#define maxit 10000
/*Precision. The program terminates when |A(n) - A(n-1)| < eps 
 where |A(n) - A(n-1)| is the Euclidean distance between the current and previous states of the matrix */
#define eps 0.02
/*To prin the resulted matrix into the file. 0 = don't print. 1 = print*/
#define printResult 0


typedef enum { Up, Down } Dir;

/*Modify the implementation of this method to set the initial values of the matrix, e.g. the boundary values*/
void initialize(double A[][m + 2]);
void print(double A[][m]);
void exchange(double A[][m + 2], int comm[], int rank);
void setComm(int rank, int comm[]);
void setOut(double out[], double A[][m + 2], Dir dir, int sender);
void getIn(double in[], double A[][m + 2], Dir dir, int receiver);
double iterate(double A[][m + 2], double mold[][m+2], int rank);
void loop(double A[][m + 2], int rank, int comm[], int* numIt);


/*The main part of the program. Controls the calculation of the next state, convergence check and communication*/
void loop(double A[][m + 2], int rank, int comm[], int* numIt)
{
	int itCount;
	/*The row that each process wil start with*/
	const int rStart = 1 + rank * rpp;
	/*The row that each process will end at*/
	const int rEnd = rStart + rpp;
	/*The number of matrix elements each process works with*/
	const int count = (rEnd - rStart) * m;

	/*The final matrix that will store the final results on process 0, excluding the boundaries*/
	double MM[n*m];
	/*The array of values that each process will send to process 0*/
	double send[count];

	for (itCount = 0; itCount < maxit; itCount++)
	{
		*numIt = itCount;
		double B[n + 2][m + 2];
		int i, j;
		for (i = 0; i < n + 2; i++)
		{
			for (j = 0; j < m + 2; j++)
			{
				B[i][j] = A[i][j];
			}
		}

		/*Iterate on the subdomain and calculate the distance between submatrices*/
		double localSum = iterate(A, B, rank);
		double globalSum;

		/*Aggregate the local distances*/
		MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if (sqrt(globalSum) < eps)
			break;
		/*Processes use this method to exchange boundary rows required for the next step of calculation*/
		 exchange(A, comm, rank);
	}

	*numIt = itCount;
	int i, j;

	/*Prepare the data to send*/
	for (i = rStart; i < rEnd; i++)
	{
		for (j = 1; j < m + 1; j++)
		{
			send[(i - rStart) * m + j - 1] = A[i][j];
		}
	}

	/*Gather the data from all processes*/
	MPI_Gather(&send, count, MPI_DOUBLE, &MM, count, MPI_DOUBLE, 0,
			MPI_COMM_WORLD);
	
	/*Print if allowed*/
	if (rank == 0 && printResult == 1)
	{
		printf("Resulted matrix \n");
		int i, j;
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < m; j++)
				printf("%3f ", MM[i * m + j]);
			printf("\n");
		}

	}
}

/*The calculation of the next state*/

double iterate(double A[][m + 2], double mold[][m+2], int rank)
{
	/*Starting row of the process*/
	const int rStart = 1 + rank * rpp;
	/*Last row of the process*/
	const int rEnd = rStart + rpp;
	int i, j;
	double sum = 0;
	/*int moldLength = m + 2;
	int moldHeight = (rEnd - rStart) + 2;
	double mold[moldLength][moldHeight];
	for (i = 0; i < moldHeight; i++)
		for (j = 0; j < moldLength; j++)
			mold[i][j] = A[i + rStart - 1][j];
	printf("Process %d mold \n", rank);
	for (i = 0; i < moldHeight; i++)
		for (j = 0; j < moldLength; j++)
			printf("%3.2f ", mold[i][j]);
	printf("\n \n \n \n \n");*/
	for (i = rStart; i < rEnd; i++)
	{
		for (j = 1; j < m + 1; j++)
		{
			/*The next state of the cell is the average of its left, top, right, bottom neighbors*/
			A[i][j] = 0.25 * (mold[i - 1][j] +
				mold[i + 1][j] +
				mold[i][j - 1] +
				mold[i][j + 1]);
			sum += (A[i][j] - mold[i][j]) * (A[i][j] - mold[i][j]);
		}
	}
	return sum;
}

void print(double A[][m])
{
	int i, j;
	for ( i = 0; i < n; i++)
	{
		for ( j = 0; j < m; j++)
			printf("%f ", A[i][j]);
		printf("\n");
	}
	printf("\n");
}

/*Initialization of the matrix
	MODIFY */

void initialize(double A[][m + 2])
{
	int i, j;
	for ( i = 0; i < n + 2; i++)
	{
		for ( j = 0; j < m + 2; j++)
		{
			A[i][j] = 0;
		}
	}

	int midX = (n + 2) / 2;
	int midY = (m + 2) / 2;
	A[midX][midY] = 100;
}

/*The exchange method*/

void exchange(double A[][m + 2], int comm[], int rank)
{
	/*
	rank = the current process
	comm[] = if comm[0] = 1(0) -> the current process can(cannot) communicate with its top neighbor
			 if comm[1] = 1(1) -> the current process can(cannot) communicate with its borrom neighbor
	*/
	/*If there is only one process involved, no need to communicate*/
	if (p == 1)
		return;
	/*Each process can communicate with at most 2 neighbors. Top and Bottom.
	  Thus, there can be at most 4 requests sent/received by a single process*/
	/*outUp[m] = the array of boundary values the current process will send to its upper neighbor*/
	double outUp[m];
	/*outDown[m] = the array of boundary values the current process will send to its down neighbor*/
	double outDown[m];
	/*inUp[m] = the array of boundary values the current process will receive from its upper neighbor*/
	double inUp[m];
	/*inDown[m] = the array of boundary values the current process will receive from its down neighbor*/
	double inDown[m];
	int partner;
	MPI_Request requests[4];
	MPI_Status status[4];
	int tag;
	int i;
	for (i = 0; i < 4; i++)
	{
		requests[i] = MPI_REQUEST_NULL;
	}

	/*Receive from up*/
	if (comm[0] == 1)
	{
		partner = rank - 1;
		tag = 0;
		MPI_Irecv(&inUp, m, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD,
			&requests[0]);
	}
	/*Receive from down*/
	if (comm[1] == 1)
	{
		partner = rank + 1;
		tag = 1;
		MPI_Irecv(&inDown, m, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD,
			&requests[1]);
	}
	/*Send to Up*/
	if (comm[0] == 1)
	{
		partner = rank - 1;
		tag = 1;
		setOut(outUp, A, Up, rank);
		MPI_Isend(&outUp, m, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD,
			&requests[2]);
	}
	/*Send to down*/
	if (comm[1] == 1)
	{
		partner = rank + 1;
		tag = 0;
		setOut(outDown, A, Down, rank);
		MPI_Isend(&outDown, m, MPI_DOUBLE, partner, tag, MPI_COMM_WORLD,
			&requests[3]);
	}

	MPI_Waitall(4, requests, status);

	if (comm[0] == 1)
	{
		getIn(inUp, A, Up, rank);
	}
	if (comm[1] == 1)
	{
		getIn(inDown, A, Down, rank);
	}
}

/*Decides with whom a given process will communicate with*/
void setComm(int rank, int comm[])
{
	/*If single processed, no communication*/
	if (p == 1)
	{
		comm[0] = 0;
		comm[1] = 0;
	}
	/*If it is first process, communicate with the bottom neighbor only*/
	else if (rank == 0)
	{
		comm[0] = 0;
		comm[1] = 1;
	}
	/*If it is the last process communicate with the top neighbor only*/
	else if (rank == p - 1)
	{
		comm[0] = 1;
		comm[1] = 0;
	}
	/*Otherwise, communicate with both neighbors*/
	else
	{
		comm[0] = 1;
		comm[1] = 1;
	}
}

/*Prepare the data to send by the process*/
void setOut(double out[], double A[][m + 2], Dir dir, int sender)
{
	/*
	dir = determines the direction where to send
	sender = the sending process
	*/
	const int rStart = 1 + sender * rpp ;
	const int rEnd = rStart + rpp - 1;

	/*If the sending direction is up, then the process must send its first working row*/
	if (dir == Up)
	{
		int i;
		for ( i = 1; i < m + 1; i++)
		{
			out[i - 1] = A[rStart][i];
		}
	}
	/*If the sending direction is down, then the process must send its last working row*/
	else
	{
		int i;
		for ( i = 1; i < m + 1; i++)
		{
			out[i - 1] = A[rEnd][i];
		}
	}
}

/*Unpacks the received data*/
void getIn(double in[], double A[][m + 2], Dir dir, int receiver)
{
	/*
	dir = determines the direction where it receives from
	receiver = the process which receives the data
	*/

	const int rStart = 1 + receiver * rpp;
	const int rEnd = rStart + rpp - 1;
	/*If the data comes from the top neighbor, then put the data in the row preceeding the first working row*/
	if (dir == Up)
	{
		int i;
		for ( i = 1; i < m + 1; i++)
		{
			A[rStart-1][i] = in[i - 1];
		}
	}
	/*If the data comes from the bottom neighbor, then put the data in the row one after the last working row*/
	else
	{
		int i;
		for ( i = 1; i < m + 1; i++)
		{
			A[rEnd+1][i] = in[i - 1];
		}
	}
}

int main(int argc, char* argv[])
{
	/*The matrix. The working matrix is  n x m, with 2 boundary rows and columns*/
	double A[n + 2][m + 2];
	int comm[2];
	int error, rank, size;
	int numIt;
	error = MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != p)
	{
		if (rank == 0)
		{
			printf("The number of processes specified does not match the actual number of processes. Terminating \n");
		}
		return;
	}

	if (n % size != 0)
	{
		if (rank == 0)
		{
			printf("The number of processes does not evenly divide the number of rows of input matrix. Terminating \n");
		}
		return;
	}

	setComm(rank, comm);
	initialize(A);

	//if (rank == 0) printf("Finished initialization \n");
	MPI_Barrier(MPI_COMM_WORLD);
	double w1 = MPI_Wtime();
	loop(A, rank, comm, &numIt);
	MPI_Barrier(MPI_COMM_WORLD);
	double w2 = MPI_Wtime();
	MPI_Finalize();
	if (rank == 0)
	{
		printf("Dimensions %d %d \n", n, m);
		printf("Total time %3f \n", w2 - w1);
		printf("Number of iterations %d \n", numIt);
		printf("Processes used %d \n", p);
	}
	
	return 0;
}
