
static char help[] = "I don't even know what this does lol.\n\n";

#include <petscksp.h>
#include <petscmath.h>

PetscScalar hat1(PetscScalar x, PetscScalar x1, PetscScalar x2){
   return (x-x1)/(x2-x1);
}

PetscScalar hat2(PetscScalar x, PetscScalar x1, PetscScalar x2){
   return (x2-x)/(x2-x1);
}

PetscScalar funcF(PetscScalar x){
   return M_PI*M_PI*PetscSinReal(M_PI*x);
}

PetscScalar simpsonHat1(PetscScalar x1, PetscScalar x2){
   PetscScalar xm,y;
   xm = (x1+x2)*0.5;
   y = (x2-x1)*(funcF(x1)*hat1(x1,x1,x2) + 4*funcF(xm)*hat1(xm,x1,x2)+ funcF(x2)*hat1(x2,x1,x2))/6;
   return y;
}

PetscScalar simpsonHat2(PetscScalar x1, PetscScalar x2){
   PetscScalar xm, y;
   xm = (x1+x2)*0.5;
   y = (x2-x1)*(funcF(x1)*hat2(x1,x1,x2) + 4*funcF(xm)*hat2(xm,x1,x2)+ funcF(x2)*hat2(x2,x1,x2) )/6;
   return y;
}

int main(int argc, char **args)
{
  Vec         x, b, u; /* approx solution, RHS, exact solution */
  Mat         A;       /* linear system matrix */
  KSP         ksp;     /* linear solver context */
  PC          pc;      /* preconditioner context */
  PetscReal   norm;    /* norm of solution error */
  PetscInt    i,j, n = 7, its;
  PetscMPIInt size;
  PetscScalar valueSquare[7][7];
  PetscScalar intitalValues[7] = {0.0,0.1,0.3,0.333,0.5,0.75,1.0};
  PetscScalar h[7]; 
  PetscInt locations[7];
  PetscScalar values[7];


  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL)); //takes the size from the command line or defaults to 10 I guess


  PetscCall(VecCreate(PETSC_COMM_SELF, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));

  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Assemble matrix
  */


  for(i=0;i<8;i++){
   h[i] = intitalValues[i+1]-intitalValues[i]; //this is the h provided in the matlab code
   locations[i] = i;
   values[i] = 0;
  }

   for(i=0;i<7;i++){
      for(j=0;j<7;j++){
         valueSquare[i][j] = 0;
      }
   }

   valueSquare[0][0] = 1;
   valueSquare[6][6] = 1;
   valueSquare[1][1] = (1/h[0]);

   for(i=1; i<5; i++){
      valueSquare[i][i] = valueSquare[i][i] + (1/h[i]);
      valueSquare[i+1][i] = valueSquare[i+1][i] - (1/h[i]);
      valueSquare[i][i+1] = valueSquare[i][i+1] - (1/h[i]);
      valueSquare[i+1][i+1] = valueSquare[i+1][i+1] + (1/h[i]);
   }

   valueSquare[n-2][n-2] = valueSquare[n-2][n-2] + (1/h[n-2]);

   /*for(i=0;i<7;i++){
      for(j=0;j<7;j++){
         PetscCall(PetscPrintf(PETSC_COMM_SELF,"%0.2f ", valueSquare[i][j]));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n"));
   }*/

   
   values[0] = 0;
   values[n-1] = 0;

   values[1] = simpsonHat1(intitalValues[0],intitalValues[1]);
   for(i=1;i<n-2;i++){
      values[i] = values[i] + simpsonHat2(intitalValues[i],intitalValues[i+1]);
      values[i+1] = values[i+1] + simpsonHat1(intitalValues[i],intitalValues[i+1]);
   }
   values[n-2] = values[n-2] + simpsonHat2(intitalValues[n-2],intitalValues[n-1]);


  for (i = 0; i < n; i++) {
   for (j = 0; j < n; j++) {
      if(valueSquare[i][j] != 0){
         PetscCall(MatSetValue(A, i, j, valueSquare[i][j], INSERT_VALUES)); //for some reason it doesn't want to let me use MatSetValues
      }
   }
  }



  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  //this is a weird way to make this matrix I got to be honest

  /*
     Set exact solution; then compute right-hand-side vector.
  */


  

  PetscCall(VecSetValues(b,n,locations,values,INSERT_VALUES));// we probably do the math on the array first, then put it into the vector

  //PetsCall(VecSetValues(u,,,,));

  //PetscCall(VecSet(u, 1.0));
  //PetscCall(MatMult(A, u, b));

   PetscCall(VecView(b, PETSC_VIEWER_STDOUT_SELF));
   PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF));


   PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n something isn't right here\n"));


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix that defines the preconditioner.
  */
  PetscCall(KSPSetOperators(ksp, A, A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp, b, x));

  /*
     View solver info; we could instead use the option -ksp_view to
     print this info to the screen at the conclusion of KSPSolve().
  */
  PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n what's going on \n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Norm of error %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));
  
  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}






/*TEST

   test:
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2_aijcusparse
      requires: cuda
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda

   test:
      suffix: 3
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3_aijcusparse
      requires: cuda
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda

   test:
      suffix: aijcusparse
      requires: cuda
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda
      output_file: output/ex1_1_aijcusparse.out

   test:
      requires: defined(PETSC_USE_SINGLE_LIBRARY)
      suffix: mpi_linear_solver_server_1
      nsize: 3
      filter: sed 's?ATOL?RTOL?g'
      args: -mpi_linear_solver_server -mpi_linear_solver_server_view -pc_type mpi -ksp_type preonly -mpi_ksp_monitor -mpi_ksp_converged_reason -mat_view -mpi_pc_type none -mpi_ksp_view -mpi_mat_view -pc_mpi_minimum_count_per_rank 5

   test:
      requires: defined(PETSC_USE_SINGLE_LIBRARY)
      suffix: mpi_linear_solver_server_2
      nsize: 3
      filter: sed 's?ATOL?RTOL?g'
      args: -mpi_linear_solver_server  -mpi_linear_solver_server_view -pc_type mpi -ksp_type preonly -mpi_ksp_monitor -mpi_ksp_converged_reason -mat_view -mpi_pc_type none -mpi_ksp_view

   test:
      requires: defined(PETSC_USE_SINGLE_LIBRARY)
      suffix: mpi_linear_solver_server_3
      nsize: 3
      filter: sed 's?ATOL?RTOL?g'
      args: -mpi_linear_solver_server  -mpi_linear_solver_server_view -pc_type mpi -ksp_type preonly -mpi_ksp_monitor -mpi_ksp_converged_reason -mat_view -mpi_pc_type none -mpi_ksp_view -mpi_mat_view -pc_mpi_always_use_server

TEST*/
