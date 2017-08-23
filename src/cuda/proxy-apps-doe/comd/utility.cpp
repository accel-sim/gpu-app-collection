#include "utility.h"

/**
 * given a struct command_t, this function will print
 * the parameters of that struct.
 **/
void printCmd(command_t *cmd) {
  printf("Command line params:\n");
  printf("  Method = %s\n"
 	 "  Input file = %s\n"
	 "  EAM flag = %d\n"
	 "  Potential dir = %s\n"
	 "  Potential name = %s\n"
	 "  Periodic = %d\n"
	 "  Use GPU = %d\n"
	 "  Number of unit cells = %d x %d x %d\n"
         "  Box factor = %g\n"
         "  Simulation temperature = %g\n"
         "  Deformation gradient = %g\n",
	 cmd->method,
	 cmd->filename,
	 cmd->doeam,
	 cmd->potdir,
	 cmd->potname,
	 cmd->periodic,
	 cmd->usegpu,
	 cmd->nx, cmd->ny, cmd->nz,
	 cmd->bf,
         cmd->temp,
         cmd->defgrad);
  return;
}

/**
 * Parse the command line parameters and fill in a
 * struct command_t.
 *
 * @param cmd a pointer to struct command_t that will be filled in
 * @param argc the number of arguments
 * @param argv the arguments array
 **/
void parseCommandLine(command_t *cmd, int argc, char **argv) {
  char spus[1024];
  char debug[1024];
  int digit_optind = 0;
  int c;
  int noperiodic=0;
  int help=0;

  // fill up cmd with defaults
#ifdef GZIPSUPPORT
  strcpy(cmd->filename,"data/8k.inp.gz");
#else
  strcpy(cmd->filename,"data/8k.inp");
#endif
  strcpy(cmd->filename,"");
  strcpy(cmd->potdir,"pots");
  strcpy(cmd->potname,"ag");
  strcpy(cmd->method,"thread_atom_warp_sync");
  cmd->doeam = 0;
  cmd->usegpu = 1; // set default to use GPU
  cmd->periodic = 1;
  cmd->nx = 20;
  cmd->ny = 20;
  cmd->nz = 20;
  cmd->bf = 1.0;
  cmd->temp = 0.0;
  cmd->defgrad = 1.0;
  cmd->nsteps = 1;
  cmd->niters = 0;
  cmd->doref = 0;


  // add arguments for processing
  addArg((char *) "help",       'h',  0,  'i',  &(help), 0,                            (char *) "print this message");
  addArg((char *) "infile",     'f',  1,  's',  cmd->filename,  sizeof(cmd->filename), (char *) "input file name");
  addArg((char *) "potdir",     'd',  1,  's',  cmd->potdir,    sizeof(cmd->potdir),   (char *) "potential directory");
  addArg((char *) "potname",    'p',  1,  's',  cmd->potname,   sizeof(cmd->potname),  (char *) "potential name");
  addArg((char *) "doeam",      'e',  0,  'i',  &(cmd->doeam),  0,                     (char *) "compute eam potentials");
  addArg((char *) "noperiodic", 'o',  0,  'i',  &noperiodic, 0,                        (char *) "do not use periodic bc");
  addArg((char *) "usegpu",     'g',  0,  'i',  &(cmd->usegpu), 0,                     (char *) "use a gpu for OpenCL");
  addArg((char *) "nx",         'x',  1,  'i',  &(cmd->nx), 0,                         (char *) "number of unit cells in x");
  addArg((char *) "ny",         'y',  1,  'i',  &(cmd->ny), 0,                         (char *) "number of unit cells in y");
  addArg((char *) "nz",         'z',  1,  'i',  &(cmd->nz), 0,                         (char *) "number of unit cells in z");
  addArg((char *) "bf",         'b',  1,  'd',  &(cmd->bf), 0,                         (char *) "box factor");
  addArg((char *) "defgrad",    's',  1,  'd',  &(cmd->defgrad), 0,                    (char *) "deformation gradient");
  addArg((char *) "temp",       't',  1,  'd',  &(cmd->temp), 0,                       (char *) "temperature");
  addArg((char *) "nsteps",     'n',  1,  'i',  &(cmd->nsteps), 0,                     (char *) "number of fraction steps per iteration");
  addArg((char *) "niters",     'i',  1,  'i',  &(cmd->niters), 0,                     (char *) "number of iterations");
  addArg((char *) "method",     'm',  1,  's',  cmd->method,    sizeof(cmd->method),   (char *) "implementation method name");
  addArg((char *) "doref",      'r',  0,  'i',  &(cmd->doref), 0,                      (char *) "run reference implementation");

  processArgs(argc,argv);
  if(help) {
    printArgs();
    freeArgs();
    exit(2);
  }
  freeArgs();
  cmd->periodic = !(noperiodic);
  return;
}

simflat_t *parse(int argc, char **argv, config &c)
{
  printf("  Precision = %s\n", (sizeof(real_t)==4?"float":"double") );

  /* get command line params */
  command_t cmd;
  parseCommandLine(&cmd, argc, argv);
  printCmd(&cmd);

  c.method = cmd.method;
  c.iters = cmd.niters;
  c.steps = cmd.nsteps;
  c.eam_flag = cmd.doeam;
  c.x = cmd.nx;
  c.y = cmd.ny;
  c.z = cmd.nz;

  /* decide whether to get LJ or EAM potentials */
  struct pmd_base_potential_t *pot;
  if (cmd.doeam) 
    pot = setEamPot(cmd.potdir, cmd.potname);
  else
    pot = (pmd_base_potential_t *) getLJPot();
  if (!pot) { printf("Unable to initialize potential\n"); exit(1); }

  /* print out simulation data */
  if (cmd.doeam) {
    eampotential_t *new_pot;
    new_pot = (eampotential_t*) pot;
    printf("EAM potential values:\n");
    printf("  cutoff = %e\n", new_pot->cutoff);
    printf("  mass = %e\n", new_pot->mass);
    cmd.lat = new_pot->lat;
    printf("  lattice = %e\n", cmd.lat);
    printf("  phi potentials = %d\n", new_pot->phi->n);
  } else {
    ljpotential_t *new_pot;
    new_pot = (ljpotential_t*) pot;
    printf("LJ potential values:\n");
    printf("  cutoff = %e\n", new_pot->cutoff);
    printf("  mass = %e\n", new_pot->mass);
    cmd.lat = 1.122462048*new_pot->cutoff ;// * 1.53;      // This is for Lennard-Jones
    printf("  lattice = %e\n", cmd.lat);
  }

  /* set initial condition */
  simflat_t *sim;
  if(strcmp(cmd.filename, "")) {
    // read from file 
    sim = fromFileASCII(cmd, pot);
    if (!sim) { printf("Input file does not existi\n"); exit(1); }
  } else {
    // init with FCC lattice 
    sim = create_fcc_lattice(cmd, pot);
  }
  printf("Initial condition set\n");

  /* setup chebyshev coefficients */
  if (cmd.doeam) {
    eampotential_t *new_pot = (eampotential_t*) pot;
    printf("Phi range = [%e .. %e]\n", new_pot->phi->x0, new_pot->phi->xn);
    sim->ch_pot = setChebPot(new_pot, 32);
  }

  printf("Total atoms = %d\n", sim->ntot);
  printf("Box factor = (%e, %e, %e)\n", sim->boxsize[0]/pot->cutoff, sim->boxsize[1]/pot->cutoff, sim->boxsize[2]/pot->cutoff);

  /* initial output for consistency check */
  reBoxAll(sim);

  return sim;
}

void CreateDevGrid(grid_t *grid_D, int cells)
{
    /** Create the device buffers to hold the grid data:
      box centers, neighbor lists info and bounds
     **/

    cudaMalloc((void**)&grid_D->r_box.x, grid_D->n_r_size);
    cudaMalloc((void**)&grid_D->r_box.y, grid_D->n_r_size);
    cudaMalloc((void**)&grid_D->r_box.z, grid_D->n_r_size);

    cudaMalloc((void**)&grid_D->neighbor_list, grid_D->n_nl_size);
    cudaMalloc((void**)&grid_D->n_neighbors, grid_D->n_n_size);
    cudaMalloc((void**)&grid_D->n_atoms, grid_D->n_n_size);

    cudaMalloc((void**)&grid_D->bounds, sizeof(real_t)*3);

        // new arrays
        cudaMalloc((void**)&grid_D->n_num_neigh, cells * sizeof(int));
        cudaMalloc((void**)&grid_D->n_neigh_boxes, cells * N_MAX_ATOMS * N_MAX_NEIGHBORS * sizeof(int));
        cudaMalloc((void**)&grid_D->n_neigh_atoms, cells * N_MAX_ATOMS * N_MAX_NEIGHBORS * sizeof(int));

        cudaMalloc((void**)&grid_D->n_list_boxes, cells * N_MAX_ATOMS * sizeof(int));
        cudaMalloc((void**)&grid_D->n_list_atoms, cells * N_MAX_ATOMS * sizeof(int));
        cudaMalloc((void**)&grid_D->itself_start_idx, cells * sizeof(int));
}

void CreateDevVec(vec_t *a_D, int array_size)
{
    cudaMalloc((void**)&a_D->x, array_size);
    cudaMalloc((void**)&a_D->y, array_size);
    cudaMalloc((void**)&a_D->z, array_size);
}

void HostEAMInit(eam_pot_t *eam_pot_H, simflat_t *sim) 
{
    /** Allocate and initialize all the EAM potential data needed **/

    int i;
    int n_v_rho;
    int n_v_phi;
    int n_v_F;

    // assign eam potential values
    printf("EAM potential sizes:\n");
    eampotential_t *new_pot;
    new_pot = (eampotential_t*) sim->pot;
    eam_pot_H->cutoff  = new_pot->cutoff;

    n_v_rho = new_pot->rho->n;
    eam_pot_H->n_p_rho_size = (6 + new_pot->rho->n)*sizeof(real_t);
    printf("  rho = %d\n", eam_pot_H->n_p_rho_size);

    n_v_phi = new_pot->phi->n;
    eam_pot_H->n_p_phi_size = (6 + new_pot->phi->n)*sizeof(real_t);
    printf("  phi = %d\n", eam_pot_H->n_p_phi_size);

    n_v_F = new_pot->f->n;
    eam_pot_H->n_p_F_size = (6 + new_pot->f->n)*sizeof(real_t);
    printf("  F = %d\n", eam_pot_H->n_p_F_size);

    eam_pot_H->rho = (real_t*)malloc(eam_pot_H->n_p_rho_size);
    eam_pot_H->phi = (real_t*)malloc(eam_pot_H->n_p_phi_size);
    eam_pot_H->F = (real_t*)malloc(eam_pot_H->n_p_F_size);
    eam_pot_H->n_values = (int*)malloc(3*sizeof(int));

    // Note the EAM array has 3 extra values to account for over/under flow
    // We also add another 3 values to store x0, xn, invDx
    eam_pot_H->rho[n_v_rho+3] = new_pot->rho->x0;
    eam_pot_H->rho[n_v_rho+4] = new_pot->rho->xn;
    eam_pot_H->rho[n_v_rho+5] = new_pot->rho->invDx;

    eam_pot_H->rho_x0 = new_pot->rho->x0;
    eam_pot_H->rho_xn = new_pot->rho->xn;
    eam_pot_H->rho_invDx = new_pot->rho->invDx;

    for (i=0;i<n_v_rho+3;i++)
    {
        eam_pot_H->rho[i] = new_pot->rho->values[i-1];
    }

    eam_pot_H->phi[n_v_phi+3] = new_pot->phi->x0;
    eam_pot_H->phi[n_v_phi+4] = new_pot->phi->xn;
    eam_pot_H->phi[n_v_phi+5] = new_pot->phi->invDx;

    eam_pot_H->phi_x0 = new_pot->phi->x0;
    eam_pot_H->phi_xn = new_pot->phi->xn;
    eam_pot_H->phi_invDx = new_pot->phi->invDx;

    for (i=0;i<n_v_phi+3;i++)
    {
        eam_pot_H->phi[i] = new_pot->phi->values[i-1];
    }

    eam_pot_H->F[n_v_F+3] = new_pot->f->x0;
    eam_pot_H->F[n_v_F+4] = new_pot->f->xn;
    eam_pot_H->F[n_v_F+5] = new_pot->f->invDx;

    for (i=0;i<n_v_F+3;i++)
    {
        eam_pot_H->F[i] = new_pot->f->values[i-1];
    }

    eam_pot_H->n_values[0] = n_v_phi;
    eam_pot_H->n_values[1] = n_v_rho;
    eam_pot_H->n_values[2] = n_v_F;
}

void HostEAMChInit(eam_ch_t *eam_ch_H, simflat_t *sim)
{
    eam_ch_H->rho = *sim->ch_pot->rho;
    eam_ch_H->phi = *sim->ch_pot->phi;
    eam_ch_H->drho = *sim->ch_pot->drho;
    eam_ch_H->dphi = *sim->ch_pot->dphi;
}

void HostLJInit(lj_pot_t *lj_pot_H, simflat_t *sim)
{
    /** Allocate and initialize all the LJ potential data needed **/

    ljpotential_t *new_pot;
    new_pot = (ljpotential_t*) sim->pot;
    lj_pot_H->sigma  = new_pot->sigma;
    lj_pot_H->epsilon  = new_pot->epsilon;
    lj_pot_H->cutoff  = new_pot->cutoff;

    printf("Using lj potential\n");
    printf("Sigma = %e\n", lj_pot_H->sigma);
    printf("Epsilon = %e\n", lj_pot_H->epsilon);
    printf("Cutoff = %e\n", lj_pot_H->cutoff);

}

void dump_sim_info(sim_t &sim_H)
{
    // atoms per box histogram
    {
    FILE *file = fopen("atoms_per_box_histogram.csv", "w");
    int hist[N_MAX_ATOMS];	// max atoms per box
    memset(hist, 0, N_MAX_ATOMS * sizeof(int));
    int ibox;
    for(ibox = 0; ibox < sim_H.n_cells; ibox++) 
      hist[sim_H.grid.n_atoms[ibox]]++;
    int i;
    for(i = 0; i < N_MAX_ATOMS; i++) 
      fprintf(file,"%i,%i,\n", i,hist[i]);
    fclose(file);
    }

    // neighbor atoms per box histogram
    {
    FILE *file = fopen("neigh_cutoff_off_histogram.csv", "w");
    int hist[N_MAX_ATOMS * N_MAX_NEIGHBORS];       // max neighbors per box
    memset(hist, 0, N_MAX_ATOMS * N_MAX_NEIGHBORS * sizeof(int));
    int ibox;
    for(ibox = 0; ibox < sim_H.n_cells; ibox++)
      hist[sim_H.grid.n_num_neigh[ibox]] += sim_H.grid.n_atoms[ibox];
    int i;
    for(i = 0; i < N_MAX_ATOMS * N_MAX_NEIGHBORS; i++)
      fprintf(file,"%i,%i,\n", i,hist[i]);
    fclose(file);
    }

    // dump neighbor atoms under cutoff 
#if 1
    {
    FILE *file = fopen("neigh_cutoff_on_histogram.csv", "w");
    int hist[N_MAX_ATOMS * N_MAX_NEIGHBORS];       // max neighbors per box
    memset(hist, 0, N_MAX_ATOMS * N_MAX_NEIGHBORS * sizeof(int));
    int ibox, iatom, jbox, jatom, nbox;
    for(ibox = 0; ibox < sim_H.n_cells; ibox++) 
      for(iatom = 0; iatom < sim_H.grid.n_atoms[ibox]; iatom++) {
        int cutoff_neigh = 0;
	for(nbox = 0; nbox < sim_H.grid.n_neighbors[ibox]; nbox++) {
          jbox = sim_H.grid.neighbor_list[ibox * N_MAX_NEIGHBORS + nbox];
          for(jatom = 0; jatom < sim_H.grid.n_atoms[jbox]; jatom++) {
            // compute box center offsets
            real_t dxbox = sim_H.grid.r_box.x[ibox] - sim_H.grid.r_box.x[jbox];
            real_t dybox = sim_H.grid.r_box.y[ibox] - sim_H.grid.r_box.y[jbox];
            real_t dzbox = sim_H.grid.r_box.z[ibox] - sim_H.grid.r_box.z[jbox];

            // correct for periodic 
            if (PERIODIC)
            {
                if (dxbox < -0.5 * sim_H.grid.bounds[0]) dxbox += sim_H.grid.bounds[0];
                else if (dxbox > 0.5 * sim_H.grid.bounds[0] ) dxbox -= sim_H.grid.bounds[0];
                if (dybox < -0.5 * sim_H.grid.bounds[1]) dybox += sim_H.grid.bounds[1];
                else if (dybox > 0.5 * sim_H.grid.bounds[1] ) dybox -= sim_H.grid.bounds[1];
                if (dzbox < -0.5 * sim_H.grid.bounds[2]) dzbox += sim_H.grid.bounds[2];
                else if (dzbox > 0.5 * sim_H.grid.bounds[2] ) dzbox -= sim_H.grid.bounds[2];
            }

                int i_particle = ibox * N_MAX_ATOMS + iatom; 
                int j_particle = jbox * N_MAX_ATOMS + jatom; 

                real_t dx = sim_H.r.x[i_particle] - sim_H.r.x[j_particle] + dxbox;
                real_t dy = sim_H.r.y[i_particle] - sim_H.r.y[j_particle] + dybox;
                real_t dz = sim_H.r.z[i_particle] - sim_H.r.z[j_particle] + dzbox;

                real_t r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < sim_H.eam_pot.cutoff * sim_H.eam_pot.cutoff) 
	      cutoff_neigh++;
          }
        }
        hist[cutoff_neigh]++;
      }
    int i;
    for(i = 0; i < N_MAX_ATOMS * N_MAX_NEIGHBORS; i++)
      fprintf(file,"%i,%i,\n", i,hist[i]);
    fclose(file);
    }
#endif

    // per-warp info 
    int warp_same_box_32 = 0;
    int warp_same_box_4 = 0;
    for (int i = 0; i < sim_H.total_atoms; i++) {
      int iatom = sim_H.grid.n_list_atoms[i];
      int ibox = sim_H.grid.n_list_boxes[i];

      int warp_size, lane_id, warp_id;
      bool all;
      
      warp_size = 32;

      lane_id = i % warp_size;
      warp_id = i / warp_size; 
      all = true;
      for (int lane = 0; lane < warp_size; lane++)
        if (sim_H.grid.n_list_boxes[warp_id * warp_size + lane] != ibox) all = false;
      if (all) warp_same_box_32++;

      warp_size = 4;

      lane_id = i % warp_size;
      warp_id = i / warp_size; 
      all = true;
      for (int lane = 0; lane < warp_size; lane++)
        if (sim_H.grid.n_list_boxes[warp_id * warp_size + lane] != ibox) all = false;
      if (all) warp_same_box_4++;
    }

    if (warp_same_box_4 % 4 != 0) printf("ERROR!\n");
    warp_same_box_4 /= 4;
    if (warp_same_box_32 % 32 != 0) printf("ERROR!\n");
    warp_same_box_32 /= 32;
 
    printf("Warp size 4: %i out of %i warps process the same set of neighbors\n", warp_same_box_4, sim_H.total_atoms / 4);
    printf("Warp size 32: %i out of %i warps process the same set of neighbors\n", warp_same_box_32, sim_H.total_atoms / 32);
}

bool is_halo(simflat_t *sim, int ibox)
{
	bool halo = false;
        int iret[3];
        iret[0] = ibox%sim->nbx[0];
        iret[1] = (ibox/sim->nbx[0])%sim->nbx[1];
        iret[2] = ibox/sim->nbx[0]/sim->nbx[1];
        if (iret[0] >= sim->nbx[0]-2) halo = true;
        if (iret[1] >= sim->nbx[1]-2) halo = true;
        if (iret[2] >= sim->nbx[2]-2) halo = true;
	return halo;
}


void create_sim_host(const config& cfg, simflat_t *sim, sim_t *sim_H)
{
    /** Allocate and initialize all the host-side simulation data needed, 
      including the appropriate potential data 
      **/
    int ibox, iatom, ioff;

    sim_H->eam_flag = cfg.eam_flag;
    sim_H->n_cells = sim->nboxes;
    sim_H->nx = sim->nbx[0];
    sim_H->ny = sim->nbx[1];
    sim_H->nz = sim->nbx[2];
    sim_H->cfac = bohr_per_atu_to_A_per_s;
    sim_H->rmass = (real_t)(amu_to_m_e)*(double)(sim->pot->mass);
    sim_H->dt = 1.0e-15*sim_H->cfac;
    printf("dt = %e\n", sim_H->dt);


    printf("Atom count per box range = ");
    int n_max_in_box = 0;
    int n_min_in_box = N_MAX_ATOMS+1;
    int box_with_max = 0;
    for (ibox=0;ibox<sim_H->n_cells;ibox++) {
        if (sim->natoms[ibox] > n_max_in_box) {
            n_max_in_box = sim->natoms[ibox];
            box_with_max = ibox;
        }
	if (sim->natoms[ibox] < n_min_in_box) {
	    n_min_in_box = sim->natoms[ibox];
 	}
    }
    printf("[%d .. %d]\n", n_min_in_box, n_max_in_box);
    sim_H->max_atoms = n_max_in_box;

    if (n_max_in_box > N_MAX_ATOMS) { 
      printf("Max # of atoms in cell = %i > N_MAX_ATOMS = %i, please update N_MAX_ATOMS in the code!\n", n_max_in_box, N_MAX_ATOMS);
      exit(1);
    }

    sim_H->array_size = N_MAX_ATOMS*sim->nboxes*sizeof(real_t);
    sim_H->grid.n_nl_size = sim->nboxes*N_MAX_NEIGHBORS*sizeof(int);
    sim_H->grid.n_n_size = sim->nboxes*sizeof(int);
    sim_H->grid.n_r_size = sim->nboxes*sizeof(real_t);

	// new arrays
	sim_H->grid.n_num_neigh = (int*)malloc(sim->nboxes * sizeof(int));
	sim_H->grid.n_neigh_boxes = (int*)malloc(sim->nboxes * N_MAX_ATOMS * N_MAX_NEIGHBORS * sizeof(int));
	sim_H->grid.n_neigh_atoms = (int*)malloc(sim->nboxes * N_MAX_ATOMS * N_MAX_NEIGHBORS * sizeof(int));

	sim_H->grid.n_list_boxes = (int*)malloc(sim->nboxes * N_MAX_ATOMS * sizeof(int));
	sim_H->grid.n_list_atoms = (int*)malloc(sim->nboxes * N_MAX_ATOMS * sizeof(int));
	sim_H->grid.itself_start_idx = (int*)malloc(sim->nboxes * sizeof(int));

    // location
    sim_H->r.x = (real_t*)malloc(sim_H->array_size);
    sim_H->r.y = (real_t*)malloc(sim_H->array_size);
    sim_H->r.z = (real_t*)malloc(sim_H->array_size);
    memset(sim_H->r.x, 0, sim_H->array_size);
    memset(sim_H->r.y, 0, sim_H->array_size);
    memset(sim_H->r.z, 0, sim_H->array_size);

    // momenta
    sim_H->p.x = (real_t*)malloc(sim_H->array_size);
    sim_H->p.y = (real_t*)malloc(sim_H->array_size);
    sim_H->p.z = (real_t*)malloc(sim_H->array_size);
    memset(sim_H->p.x, 0, sim_H->array_size);
    memset(sim_H->p.y, 0, sim_H->array_size);
    memset(sim_H->p.z, 0, sim_H->array_size);

    // forces
    sim_H->f.x = (real_t*)malloc(sim_H->array_size);
    sim_H->f.y = (real_t*)malloc(sim_H->array_size);
    sim_H->f.z = (real_t*)malloc(sim_H->array_size);
    memset(sim_H->f.x, 0, sim_H->array_size);
    memset(sim_H->f.y, 0, sim_H->array_size);
    memset(sim_H->f.z, 0, sim_H->array_size);

    // box locations
    sim_H->grid.r_box.x = (real_t*)malloc(sim_H->grid.n_r_size);
    sim_H->grid.r_box.y = (real_t*)malloc(sim_H->grid.n_r_size);
    sim_H->grid.r_box.z = (real_t*)malloc(sim_H->grid.n_r_size);

    sim_H->grid.neighbor_list = (int*)malloc(sim_H->grid.n_nl_size);
    sim_H->grid.n_neighbors = (int*)malloc(sim_H->grid.n_n_size);
    sim_H->grid.n_atoms = (int*)malloc(sim_H->grid.n_n_size);
    sim_H->grid.bounds = (real_t*)malloc(3*sizeof(real_t));

    // mass, energy
    sim_H->m = (real_t*)malloc(sim_H->array_size);
    sim_H->e = (real_t*)malloc(sim_H->array_size);

    if(sim_H->eam_flag) {
        sim_H->fi = (real_t*)malloc(sim_H->array_size);
        sim_H->rho = (real_t*)malloc(sim_H->array_size);

        HostEAMInit(&sim_H->eam_pot, sim);
        HostEAMChInit(&sim_H->ch_pot, sim);
    } else {
        HostLJInit(&sim_H->lj_pot, sim);
    }

    sim_H->total_atoms = 0;
    for(ibox=0;ibox<sim->nboxes;ibox++) {

	bool halo = is_halo(sim, ibox);

        int* nbrBoxes;
	if (!halo)
          nbrBoxes = getNeighborBoxes(sim,ibox);

        sim_H->grid.n_atoms[ibox] = sim->natoms[ibox];

        sim_H->grid.r_box.x[ibox] = sim->dcenter[ibox][0];
        sim_H->grid.r_box.y[ibox] = sim->dcenter[ibox][1];
        sim_H->grid.r_box.z[ibox] = sim->dcenter[ibox][2];

        int j;
	if (!halo) {
          sim_H->grid.n_neighbors[ibox] = nbrBoxes[-1];
          for (j=0;j<sim_H->grid.n_neighbors[ibox];j++) {
            sim_H->grid.neighbor_list[N_MAX_NEIGHBORS*ibox + j] = nbrBoxes[j];
          }
 	}

        for(iatom=0;iatom<sim->natoms[ibox];iatom++) {

            ioff = ibox*N_MAX_ATOMS + iatom;

            sim_H->r.x[ioff] = sim->r[ioff][0];
            sim_H->r.y[ioff] = sim->r[ioff][1];
            sim_H->r.z[ioff] = sim->r[ioff][2];

            sim_H->p.x[ioff] = sim->p[ioff][0];
            sim_H->p.y[ioff] = sim->p[ioff][1];
            sim_H->p.z[ioff] = sim->p[ioff][2];

            sim_H->f.x[ioff] = sim->f[ioff][0];
            sim_H->f.y[ioff] = sim->f[ioff][1];
            sim_H->f.z[ioff] = sim->f[ioff][2];

	    if (!halo) {
              sim_H->grid.n_list_boxes[sim_H->total_atoms] = ibox;
	      sim_H->grid.n_list_atoms[sim_H->total_atoms] = iatom;
	      sim_H->total_atoms++;
	    }
        }

        for (j=0;j<3;j++) {
            sim_H->grid.bounds[j] = sim->bounds[j];
        }

	// new arrays
	sim_H->grid.n_num_neigh[ibox] = 0;
	if (!halo)
	  for (j=0;j<sim_H->grid.n_neighbors[ibox];j++) 
            sim_H->grid.n_num_neigh[ibox] += sim->natoms[nbrBoxes[j]];

	int global_id = ibox * N_MAX_ATOMS * N_MAX_NEIGHBORS;
	int k;

        // current box in the neighbor list has index 13
        int itself = 13;

        // store itself neighbor starting location
        int itself_start_idx = 0;

	if (!halo) {
	  for (j = 0; j < 13; j++) 
	    itself_start_idx += sim->natoms[nbrBoxes[j]];
	  sim_H->grid.itself_start_idx[ibox] = itself_start_idx;
        
	  // add all neighbors
	  for (j=0;j<sim_H->grid.n_neighbors[ibox];j++) { 
		for(k = 0; k < sim->natoms[nbrBoxes[j]]; k++ ) {
			sim_H->grid.n_neigh_boxes[global_id] = nbrBoxes[j];
			sim_H->grid.n_neigh_atoms[global_id] = k;
			global_id++;
		}
	  }
	}
    }
}

void create_sim_dev(sim_t &sim_H, sim_t *sim_D)
{
    /** Allocate all the device-side arrays needed for the simulation **/

    sim_D->dt = sim_H.dt;
    sim_D->rmass = sim_H.rmass;
    sim_D->max_atoms = sim_H.max_atoms;
    sim_D->total_atoms = sim_H.total_atoms;
    sim_D->n_cells = sim_H.n_cells;
    sim_D->nx = sim_H.nx;
    sim_D->ny = sim_H.ny;
    sim_D->nz = sim_H.nz;
    sim_D->grid.n_r_size = sim_H.grid.n_r_size;
    sim_D->grid.n_nl_size = sim_H.grid.n_nl_size;
    sim_D->grid.n_n_size = sim_H.grid.n_n_size;
    sim_D->array_size = sim_H.array_size;

    // positions
    CreateDevVec(&sim_D->r, sim_H.array_size);
    CreateDevVec(&sim_D->p, sim_H.array_size);
    CreateDevVec(&sim_D->f, sim_H.array_size);

    // particle mass
    cudaMalloc((void**)&sim_D->m, sim_H.array_size);

    // particle energy
    cudaMalloc((void**)&sim_D->e, sim_H.array_size);

    CreateDevGrid(&sim_D->grid, sim_H.n_cells);

    if (sim_H.eam_flag){
        cudaMalloc((void**)&sim_D->fi, sim_H.array_size);
        cudaMalloc((void**)&sim_D->rho, sim_H.array_size);

        cudaMalloc((void**)&sim_D->eam_pot.rho, sim_H.eam_pot.n_p_rho_size);
        cudaMalloc((void**)&sim_D->eam_pot.phi, sim_H.eam_pot.n_p_phi_size);
        cudaMalloc((void**)&sim_D->eam_pot.F, sim_H.eam_pot.n_p_F_size);

        cudaMalloc((void**)&sim_D->eam_pot.n_values, sizeof(int)*3);

        // add this here to make passing arguments to kernels easier
        sim_D->eam_pot.cutoff = sim_H.eam_pot.cutoff;

	sim_D->eam_pot.rho_x0 = sim_H.eam_pot.rho_x0;
	sim_D->eam_pot.rho_xn = sim_H.eam_pot.rho_xn;
	sim_D->eam_pot.rho_invDx = sim_H.eam_pot.rho_invDx;

	sim_D->eam_pot.phi_x0 = sim_H.eam_pot.phi_x0;
	sim_D->eam_pot.phi_xn = sim_H.eam_pot.phi_xn;
	sim_D->eam_pot.phi_invDx = sim_H.eam_pot.phi_invDx;
	
        // init cheby
        sim_D->ch_pot.rho = sim_H.ch_pot.rho;
        sim_D->ch_pot.phi = sim_H.ch_pot.phi;
        sim_D->ch_pot.drho = sim_H.ch_pot.drho;
        sim_D->ch_pot.dphi = sim_H.ch_pot.dphi;

        // alloc cheby
        cudaMalloc((void**)&sim_D->ch_pot.rho.values, (sim_H.ch_pot.rho.n + 3) * sizeof(real_t)); 
        cudaMalloc((void**)&sim_D->ch_pot.phi.values, (sim_H.ch_pot.phi.n + 3) * sizeof(real_t)); 
        cudaMalloc((void**)&sim_D->ch_pot.drho.values, (sim_H.ch_pot.drho.n + 3) * sizeof(real_t));
        cudaMalloc((void**)&sim_D->ch_pot.dphi.values, (sim_H.ch_pot.dphi.n + 3) * sizeof(real_t)); 
    } else {
        sim_D->lj_pot.cutoff = sim_H.lj_pot.cutoff;
        sim_D->lj_pot.sigma = sim_H.lj_pot.sigma;
        sim_D->lj_pot.epsilon = sim_H.lj_pot.epsilon;
    }
    printf("Device memory allocated\n");

}

void GetVec(vec_t a_D,
        vec_t a_H,
        int array_size)
{
    cudaMemcpy(a_H.x, a_D.x, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_H.y, a_D.y, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_H.z, a_D.z, array_size, cudaMemcpyDeviceToHost);
}

void PutVec(
        vec_t a_H,
        vec_t a_D,
        int array_size)
{
    cudaMemcpy(a_D.x, a_H.x, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_D.y, a_H.y, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_D.z, a_H.z, array_size, cudaMemcpyHostToDevice);   
}

void GetVector(
	real_t* ax_D, real_t* ay_D, real_t* az_D,
        real_t* ax_H, real_t* ay_H, real_t* az_H,
        int array_size)
{
    cudaMemcpy(ax_H, ax_D, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ay_H, ay_D, array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(az_H, az_D, array_size, cudaMemcpyDeviceToHost);
}

void PutVector(
        real_t* ax_H, real_t* ay_H, real_t* az_H,
        real_t* ax_D, real_t* ay_D, real_t* az_D,
        int array_size)
{
    cudaMemcpy(ax_D, ax_H, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ay_D, ay_H, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(az_D, az_H, array_size, cudaMemcpyHostToDevice);
}

void PutGrid(grid_t grid_H, grid_t grid_D, int cells)
{
    PutVec(grid_H.r_box, grid_D.r_box, grid_H.n_r_size);

    cudaMemcpy(grid_D.n_neighbors, grid_H.n_neighbors, grid_H.n_n_size, cudaMemcpyHostToDevice);
    cudaMemcpy(grid_D.n_atoms, grid_H.n_atoms, grid_H.n_n_size, cudaMemcpyHostToDevice);
    cudaMemcpy(grid_D.neighbor_list, grid_H.neighbor_list, grid_H.n_nl_size, cudaMemcpyHostToDevice);

    cudaMemcpy(grid_D.bounds, grid_H.bounds, sizeof(real_t)*3, cudaMemcpyHostToDevice);

	cudaMemcpy(grid_D.n_num_neigh, grid_H.n_num_neigh, sizeof(int) * cells, cudaMemcpyHostToDevice);
	cudaMemcpy(grid_D.n_neigh_boxes, grid_H.n_neigh_boxes, sizeof(int) * cells * N_MAX_ATOMS * N_MAX_NEIGHBORS, cudaMemcpyHostToDevice);
	cudaMemcpy(grid_D.n_neigh_atoms, grid_H.n_neigh_atoms, sizeof(int) * cells * N_MAX_ATOMS * N_MAX_NEIGHBORS, cudaMemcpyHostToDevice);

	cudaMemcpy(grid_D.n_list_boxes, grid_H.n_list_boxes, sizeof(int) * cells * N_MAX_ATOMS, cudaMemcpyHostToDevice);
	cudaMemcpy(grid_D.n_list_atoms, grid_H.n_list_atoms, sizeof(int) * cells * N_MAX_ATOMS, cudaMemcpyHostToDevice);
	cudaMemcpy(grid_D.itself_start_idx, grid_H.itself_start_idx, sizeof(int) * cells, cudaMemcpyHostToDevice);
}

void PutEamPot(eam_pot_t eam_pot_H, eam_pot_t eam_pot_D)
{
    cudaMemcpy(eam_pot_D.rho, eam_pot_H.rho, eam_pot_H.n_p_rho_size, cudaMemcpyHostToDevice);
    cudaMemcpy(eam_pot_D.phi, eam_pot_H.phi, eam_pot_H.n_p_phi_size, cudaMemcpyHostToDevice);
    cudaMemcpy(eam_pot_D.F, eam_pot_H.F, eam_pot_H.n_p_F_size, cudaMemcpyHostToDevice);
    cudaMemcpy(eam_pot_D.n_values, eam_pot_H.n_values, sizeof(int)*3, cudaMemcpyHostToDevice);
}

void PutEamChPot(eam_ch_t eam_ch_H, eam_ch_t eam_ch_D)
{
    cudaMemcpy(eam_ch_D.rho.values, eam_ch_H.rho.values, (eam_ch_H.rho.n + 3) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(eam_ch_D.phi.values, eam_ch_H.phi.values, (eam_ch_H.phi.n + 3) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(eam_ch_D.drho.values, eam_ch_H.drho.values, (eam_ch_H.drho.n + 3) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(eam_ch_D.dphi.values, eam_ch_H.dphi.values, (eam_ch_H.dphi.n + 3) * sizeof(real_t), cudaMemcpyHostToDevice);
}

void fill_sim_dev(sim_t &sim_H, sim_t &sim_D)
{
      /** Copy all the host-side simulation data to the corresponding device arrays **/

    // copy the input arrays to the device
    // positions

    PutVec(sim_H.r, sim_D.r, sim_H.array_size);
    PutVec(sim_H.p, sim_D.p, sim_H.array_size);
    PutVec(sim_H.f, sim_D.f, sim_H.array_size);

    // mass
    cudaMemcpy(sim_D.m, sim_H.m, sim_H.array_size, cudaMemcpyHostToDevice);

    // simulation data
    PutGrid(sim_H.grid, sim_D.grid, sim_H.n_cells);

    if(sim_H.eam_flag) {
        PutEamPot(sim_H.eam_pot, sim_D.eam_pot);
        PutEamChPot(sim_H.ch_pot, sim_D.ch_pot);
    }
    printf("Data copied to device\n");
}

void destroy_sim_dev(sim_t &sim_D)
{
    cudaFree(sim_D.r.x);
    cudaFree(sim_D.r.y);
    cudaFree(sim_D.r.z);

    cudaFree(sim_D.p.x);
    cudaFree(sim_D.p.y);
    cudaFree(sim_D.p.z);

    cudaFree(sim_D.f.x);
    cudaFree(sim_D.f.y);
    cudaFree(sim_D.f.z);

    cudaFree(sim_D.m);
    cudaFree(sim_D.e);

    if(sim_D.eam_flag) {
        cudaFree(sim_D.rho);
        cudaFree(sim_D.fi);
    }
}

void compute_energy(simflat_t *sim, sim_t &sim_D)
{
/**
      Copy the array of particle energies from device to host. 
      The total energy is summed and returned in the sim_H.energy variable
     **/

    int ibox, iatom;
    double local_e;

    real_t *energy = (real_t*)malloc(sim_D.array_size);
    int *n_atoms = (int*)malloc(sim_D.n_cells * sizeof(int));
    cudaMemcpy(energy, sim_D.e, sim_D.array_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(n_atoms, sim_D.grid.n_atoms, sim_D.n_cells * sizeof(int), cudaMemcpyDeviceToHost);

    local_e = 0.0;
    for (ibox=0;ibox<sim_D.n_cells;ibox++) {
	if (is_halo(sim, ibox)) continue;
        for (iatom=0;iatom<n_atoms[ibox];iatom++) {
            local_e += energy[ibox*N_MAX_ATOMS + iatom];
        }
    }
    sim_D.energy = (real_t)local_e;
}
