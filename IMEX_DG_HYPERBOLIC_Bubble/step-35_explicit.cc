// @sect3{Include files}

// We start by including all the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/base/timer.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

// Finally this is as in all previous programs:
namespace Step35 {
  using namespace dealii;

  // @sect3{Run time parameters}
  //
  // Since our method has several parameters that can be fine-tuned we put them
  // into an external file, so that they can be determined at run-time.
  //
  namespace RunTimeParameters {
    class Data_Storage {
    public:
      Data_Storage();

      void read_data(const std::string& filename);

      double initial_time;
      double final_time;

      double Reynolds;
      double Prandtl;
      double Mach;
      double Froude;
      double dt;
      double CFL;

      unsigned int n_global_refines;
      unsigned int n_cells;
      unsigned int max_loc_refinements;
      unsigned int min_loc_refinements;

      unsigned int vel_max_iterations;
      unsigned int vel_Krylov_size;
      unsigned int vel_off_diagonals;
      unsigned int vel_update_prec;
      double       vel_eps;
      double       vel_diag_strength;

      bool         verbose;
      unsigned int output_interval;

      std::string dir;

      unsigned int refinement_iterations;

    protected:
      ParameterHandler prm;
    };

    // In the constructor of this class we declare all the parameters.
    Data_Storage::Data_Storage(): initial_time(0.0),
                                  final_time(1.0),
                                  Reynolds(1.0),
                                  Prandtl(1.0),
                                  Mach(1.0),
                                  Froude(1.0),
                                  dt(5e-4),
                                  CFL(1.0),
                                  n_global_refines(0),
                                  n_cells(0),
                                  max_loc_refinements(0),
                                  min_loc_refinements(0),
                                  vel_max_iterations(1000),
                                  vel_Krylov_size(30),
                                  vel_off_diagonals(60),
                                  vel_update_prec(15),
                                  vel_eps(1e-12),
                                  vel_diag_strength(0.01),
                                  verbose(true),
                                  output_interval(15),
                                  refinement_iterations(0) {
      prm.enter_subsection("Physical data");
      {
        prm.declare_entry("initial_time",
                          "0.0",
                          Patterns::Double(0.0),
                          " The initial time of the simulation. ");
        prm.declare_entry("final_time",
                          "1.0",
                          Patterns::Double(0.0),
                          " The final time of the simulation. ");
        prm.declare_entry("Reynolds",
                          "1.0",
                          Patterns::Double(0.0),
                          " The Reynolds number. ");
        prm.declare_entry("Prandtl",
                          "1.0",
                          Patterns::Double(0.0),
                          " The Prandtl number. ");
        prm.declare_entry("Mach",
                          "1.0",
                          Patterns::Double(0.0),
                          " The Mach number. ");
        prm.declare_entry("Froude",
                          "1.0",
                          Patterns::Double(0.0),
                          " The Froude number. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        prm.declare_entry("dt",
                          "5e-4",
                          Patterns::Double(0.0),
                          " The time step size. ");
        prm.declare_entry("CFL",
                          "1.0",
                          Patterns::Double(0.0),
                          " The Courant number. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        prm.declare_entry("n_of_refines",
                          "3",
                          Patterns::Integer(1, 15),
                          " The number of global refinements we want for the mesh. ");
        prm.declare_entry("n_of_cells",
                          "100",
                          Patterns::Integer(1, 1500),
                          " The number of cells we want on each direction of the mesh. ");
        prm.declare_entry("max_loc_refinements",
                          "4",
                           Patterns::Integer(1, 10),
                           " The number of maximum local refinements. ");
        prm.declare_entry("min_loc_refinements",
                          "2",
                           Patterns::Integer(1, 10),
                           " The number of minimum local refinements. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        prm.declare_entry("max_iterations",
                          "1000",
                          Patterns::Integer(1, 30000),
                          " The maximal number of iterations GMRES must make. ");
        prm.declare_entry("eps",
                          "1e-12",
                          Patterns::Double(0.0),
                          " The stopping criterion. ");
        prm.declare_entry("Krylov_size",
                          "30",
                          Patterns::Integer(1),
                          " The size of the Krylov subspace to be used. ");
        prm.declare_entry("off_diagonals",
                          "60",
                          Patterns::Integer(0),
                          " The number of off-diagonal elements ILU must "
                          "compute. ");
        prm.declare_entry("diag_strength",
                          "0.01",
                          Patterns::Double(0.0),
                          " Diagonal strengthening coefficient. ");
        prm.declare_entry("update_prec",
                          "15",
                          Patterns::Integer(1),
                          " This number indicates how often we need to "
                          "update the preconditioner");
      }
      prm.leave_subsection();

      prm.declare_entry("refinement_iterations",
                        "0",
                        Patterns::Integer(0),
                        " This number indicates how often we need to "
                        "refine the mesh");

      prm.declare_entry("saving directory", "SimTest");

      prm.declare_entry("verbose",
                        "true",
                        Patterns::Bool(),
                        " This indicates whether the output of the solution "
                        "process should be verbose. ");

      prm.declare_entry("output_interval",
                        "1",
                        Patterns::Integer(1),
                        " This indicates between how many time steps we print "
                        "the solution. ");
    }

    // Function to read all declared parameters in the constructor
    void Data_Storage::read_data(const std::string& filename) {
      std::ifstream file(filename);
      AssertThrow(file, ExcFileNotOpen(filename));

      prm.parse_input(file);

      prm.enter_subsection("Physical data");
      {
        initial_time = prm.get_double("initial_time");
        final_time   = prm.get_double("final_time");
        Reynolds     = prm.get_double("Reynolds");
        Prandtl      = prm.get_double("Prandtl");
        Mach         = prm.get_double("Mach");
        Froude       = prm.get_double("Froude");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        dt = prm.get_double("dt");
        CFL = prm.get_double("CFL");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        n_global_refines = prm.get_integer("n_of_refines");
        n_cells = prm.get_integer("n_of_cells");
        max_loc_refinements = prm.get_integer("max_loc_refinements");
        min_loc_refinements = prm.get_integer("min_loc_refinements");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        vel_max_iterations = prm.get_integer("max_iterations");
        vel_eps            = prm.get_double("eps");
        vel_Krylov_size    = prm.get_integer("Krylov_size");
        vel_off_diagonals  = prm.get_integer("off_diagonals");
        vel_diag_strength  = prm.get_double("diag_strength");
        vel_update_prec    = prm.get_integer("update_prec");
      }
      prm.leave_subsection();

      dir = prm.get("saving directory");

      refinement_iterations = prm.get_integer("refinement_iterations");

      verbose = prm.get_bool("verbose");

      output_interval = prm.get_integer("output_interval");
    }

  } // namespace RunTimeParameters


  // @sect3{Equation data}

  // In the next namespace, we declare the initial and boundary conditions
  //
  namespace EquationData {
    static const unsigned int degree_T = 1;
    static const unsigned int degree_rho = 1;
    static const unsigned int degree_u = 1;

    static const double Cp_Cv = 1.4;

    // With this class defined, we declare class that describes the boundary
    // conditions for velocity:
    template<int dim>
    class Velocity: public Function<dim> {
    public:
      Velocity(const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;

      virtual void vector_value(const Point<dim>& p,
                                Vector<double>&   values) const override;
    };


    template<int dim>
    Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time) {}


    template<int dim>
    double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
      AssertIndexRange(component, 2);

      return 0.0;
    }


    template<int dim>
    void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
      Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
      for(unsigned int i = 0; i < dim; ++i)
        values[i] = value(p, i);
    }


    // We do the same for the pressure (since it is a scalar field) we can derive
    // directly from the deal.II built-in class Function
    template<int dim>
    class Energy: public Function<dim> {
    public:
      Energy(const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;
    };


    template<int dim>
    Energy<dim>::Energy(const double initial_time): Function<dim>(1, initial_time) {}


    template<int dim>
    double Energy<dim>::value(const Point<dim>&  p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);

      const double x0 = 500.0;
      const double y0 = 260.0;
      const double r  = std::sqrt((p[0] - x0)*(p[0] - x0) + (p[1] - y0)*(p[1] - y0));
      const double a = 50.0;
      const double s = 100.0;
      const double A = 0.5;

      const double theta_prime = A*(r <= a) + A*std::exp(-(r-a)*(r-a)/(s*s))*(r > a);
      const double theta = 300.0 + theta_prime;

      const double Cp = EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*287.05;
      const double pi_e = 1.0 - 9.81*p[1]/(Cp*300.0);
      const double rho_bar = 100000.0/(287.05*300.0)*std::pow(pi_e, 1.0/(EquationData::Cp_Cv - 1.0));
      const double rho = rho_bar*(1.0 - theta_prime/theta);

      return 1.0/(EquationData::Cp_Cv - 1.0)*100000.0*std::pow(287.05*rho*theta/100000.0, EquationData::Cp_Cv);
    }


    // We do the same for the density (since it is a scalar field) we can derive
    // directly from the deal.II built-in class Function
    template<int dim>
    class Density: public Function<dim> {
    public:
      Density(const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;
    };


    template<int dim>
    Density<dim>::Density(const double initial_time): Function<dim>(1, initial_time) {}


    template<int dim>
    double Density<dim>::value(const Point<dim>&  p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);

      const double x0 = 500.0;
      const double y0 = 260.0;
      const double r  = std::sqrt((p[0] - x0)*(p[0] - x0) + (p[1] - y0)*(p[1] - y0));
      const double a = 50.0;
      const double s = 100.0;
      const double A = 0.5;

      const double theta_prime = A*(r <= a) + A*std::exp(-(r-a)*(r-a)/(s*s))*(r > a);
      const double theta = 300.0 + theta_prime;

      const double Cp = EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*287.05;
      const double pi_e = 1.0 - 9.81*p[1]/(Cp*300.0);
      const double rho_bar = 100000.0/(287.05*300.0)*std::pow(pi_e, 1.0/(EquationData::Cp_Cv - 1.0));

      return rho_bar*(1.0 - theta_prime/theta);
    }

  } // namespace EquationData


  // @sect3{ <code>HYPERBOLICOperator::HYPERBOLICOperator</code> }
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  class HYPERBOLICOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    HYPERBOLICOperator();

    HYPERBOLICOperator(RunTimeParameters::Data_Storage& data);

    void set_dt(const double time_step);

    void set_Mach(const double Ma_);

    void set_Froude(const double Fr_);

    void set_HYPERBOLIC_stage(const unsigned int stage);

    void set_NS_stage(const unsigned int stage);

    void set_rho_for_fixed(const Vec& src);

    void set_pres_fixed(const Vec& src);

    void vmult_rhs_rho_projection(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_momentum_update(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_energy(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_velocity_projection(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_pressure_projection(Vec& dst, const std::vector<Vec>& src) const;

    virtual void compute_diagonal() override {}

  protected:
    double       Ma;
    double       Fr;
    double       dt;

    double       gamma;
    double       a21;
    double       a22;
    double       a31;
    double       a32;
    double       a33;
    double       a21_tilde;
    double       a22_tilde;
    double       a31_tilde;
    double       a32_tilde;
    double       a33_tilde;

    unsigned int HYPERBOLIC_stage;
    unsigned int NS_stage;

    virtual void apply_add(Vec& dst, const Vec& src) const override;

  private:
    Vec rho_for_fixed, pres_fixed;

    void assemble_rhs_cell_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                               Vec&                                         dst,
                                               const std::vector<Vec>&                      src,
                                               const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                               Vec&                                         dst,
                                               const std::vector<Vec>&                      src,
                                               const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                                   Vec&                                         dst,
                                                   const std::vector<Vec>&                      src,
                                                   const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_rhs_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                    Vec&                                         dst,
                                                    const std::vector<Vec>&                      src,
                                                    const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_rhs_cell_term_energy(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_energy(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_energy(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_energy(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const Vec&                                   src,
                                   const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_rhs_cell_term_velocity_projection(const MatrixFree<dim, Number>&               data,
                                                    Vec&                                         dst,
                                                    const std::vector<Vec>&                      src,
                                                    const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_cell_term_velocity_projection(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const Vec&                                   src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_rhs_cell_term_pressure_projection(const MatrixFree<dim, Number>&               data,
                                                    Vec&                                         dst,
                                                    const std::vector<Vec>&                      src,
                                                    const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_cell_term_pressure_projection(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const Vec&                                   src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;
  };


  // Default constructor
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u, n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  HYPERBOLICOperator(): MatrixFreeOperators::Base<dim, Vec>(), Ma(), Fr(), dt(), gamma(2.0 - std::sqrt(2.0)),
                        a21(gamma), a22(0.0), a31((2.0*gamma - 1.0)/6.0), a32((7.0 - 2.0*gamma)/6.0), a33(0.0),
                        a21_tilde(0.5*gamma), a22_tilde(0.5*gamma), a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                        a33_tilde(1.0 - std::sqrt(2)/4.0), HYPERBOLIC_stage(1), NS_stage(1) {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u, n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  HYPERBOLICOperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(),
                                                             Ma(data.Mach), Fr(data.Froude), dt(data.dt), gamma(2.0 - std::sqrt(2.0)),
                                                             a21(gamma), a22(0.0), a31((2.0*gamma - 1.0)/6.0),
                                                             a32((7.0 - 2.0*gamma)/6.0), a33(0.0),
                                                             a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                                                             a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                                                             a33_tilde(1.0 - std::sqrt(2)/4.0), HYPERBOLIC_stage(1), NS_stage(1) {}


  // Setter of time-step
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of Mach number
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_Mach(const double Ma_) {
    Ma = Ma_;
  }


  // Setter of Froude number
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_Froude(const double Fr_) {
    Fr = Fr_;
  }

  // Setter of HYPERBOLIC stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_HYPERBOLIC_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());
    HYPERBOLIC_stage = stage;
  }


  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 5);
    Assert(stage > 0, ExcInternalError());
    NS_stage = stage;
  }


  // Setter of density for fixed point
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_rho_for_fixed(const Vec& src) {
    rho_for_fixed = src;
    rho_for_fixed.update_ghost_values();
  }


  // Setter of pressure for fixed point
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_pres_fixed(const Vec& src) {
    pres_fixed = src;
    pres_fixed.update_ghost_values();
  }


  // Assemble rhs cell term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi(data, 2), phi_rho_old(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi_rho_old.get_value(q), q);
          phi.submit_gradient(a21*dt*phi_u_old.get_value(q), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi(data, 2), phi_rho_old(data, 2), phi_rho_tmp_2(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0), phi_u_tmp_2(data, 0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi_rho_old.get_value(q), q);
          phi.submit_gradient(a31*dt*phi_u_old.get_value(q) +
                              a32*dt*phi_u_tmp_2.get_value(q), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_p(data, true, 2), phi_m(data, false, 2),
                                                                         phi_rho_old_p(data, true, 2), phi_rho_old_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0), phi_u_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old_p(data, true, 1), phi_E_old_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_E_old_p.reinit(face);
        phi_E_old_p.gather_evaluate(src[2], true, false);
        phi_E_old_m.reinit(face);
        phi_E_old_m.gather_evaluate(src[2], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus     = phi_p.get_normal_vector(q);

          const auto& avg_flux   = 0.5*(phi_u_old_p.get_value(q) + phi_u_old_m.get_value(q));
          const auto& pres_old_p = (EquationData::Cp_Cv - 1.0)*
                                   (phi_E_old_p.get_value(q) -
                                    0.5/phi_rho_old_p.get_value(q)*
                                    scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q)));
          const auto& pres_old_m = (EquationData::Cp_Cv - 1.0)*
                                   (phi_E_old_m.get_value(q) -
                                    0.5/phi_rho_old_m.get_value(q)*
                                    scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q)));
          const auto  lambda     = std::max(std::sqrt(scalar_product(phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q),
                                                                     phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q))) +
                                            1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_p/phi_rho_old_p.get_value(q)),
                                            std::sqrt(scalar_product(phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q),
                                                                     phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q))) +
                                            1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_m/phi_rho_old_m.get_value(q)));
          const auto& jump_rho_old = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          phi_p.submit_value(-a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho_old), q);
          phi_m.submit_value(a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho_old), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_p(data, true, 2), phi_m(data, false, 2),
                                                                         phi_rho_old_p(data, true, 2), phi_rho_old_m(data, false, 2),
                                                                         phi_rho_tmp_2_p(data, true, 2), phi_rho_tmp_2_m(data, false ,2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0), phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0), phi_u_tmp_2_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old_p(data, true, 1), phi_E_old_m(data, false, 1),
                                                                     phi_E_tmp_2_p(data, true, 1), phi_E_tmp_2_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_E_old_p.reinit(face);
        phi_E_old_p.gather_evaluate(src[2], true, false);
        phi_E_old_m.reinit(face);
        phi_E_old_m.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_E_tmp_2_p.reinit(face);
        phi_E_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_E_tmp_2_m.reinit(face);
        phi_E_tmp_2_m.gather_evaluate(src[5], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& avg_flux_old   = 0.5*(phi_u_old_p.get_value(q) + phi_u_old_m.get_value(q));
          const auto& pres_old_p     = (EquationData::Cp_Cv - 1.0)*
                                       (phi_E_old_p.get_value(q) -
                                        0.5/phi_rho_old_p.get_value(q)*
                                        scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q)));
          const auto& pres_old_m     = (EquationData::Cp_Cv - 1.0)*
                                       (phi_E_old_m.get_value(q) -
                                        0.5/phi_rho_old_m.get_value(q)*
                                        scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q)));
          const auto  lambda_old     = std::max(std::sqrt(scalar_product(phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q),
                                                                         phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_p/phi_rho_old_p.get_value(q)),
                                                std::sqrt(scalar_product(phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q),
                                                                         phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_m/phi_rho_old_m.get_value(q)));
          const auto& jump_rho_old   = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          const auto& avg_flux_tmp_2 = 0.5*(phi_u_tmp_2_p.get_value(q) + phi_u_tmp_2_m.get_value(q));
          const auto& pres_tmp_2_p   = (EquationData::Cp_Cv - 1.0)*
                                       (phi_E_tmp_2_p.get_value(q) -
                                        0.5/phi_rho_tmp_2_p.get_value(q)*
                                        scalar_product(phi_u_tmp_2_p.get_value(q), phi_u_tmp_2_p.get_value(q)));
          const auto& pres_tmp_2_m   = (EquationData::Cp_Cv - 1.0)*
                                       (phi_E_tmp_2_m.get_value(q) -
                                        0.5/phi_rho_tmp_2_m.get_value(q)*
                                        scalar_product(phi_u_tmp_2_m.get_value(q), phi_u_tmp_2_m.get_value(q)));
          const auto  lambda_tmp_2   = std::max(std::sqrt(scalar_product(phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q),
                                                                         phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_p/phi_rho_tmp_2_p.get_value(q)),
                                                std::sqrt(scalar_product(phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q),
                                                                         phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_m/phi_rho_tmp_2_m.get_value(q)));
          const auto& jump_rho_tmp_2 = phi_rho_tmp_2_p.get_value(q) - phi_rho_tmp_2_m.get_value(q);

          phi_p.submit_value(-a31*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old)
                             -a32*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
          phi_m.submit_value(a31*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old) +
                             a32*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const std::vector<Vec>&                      src,
                                            const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi(data, true, 2), phi_rho_old(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(face);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          auto        rho_old_D        = phi_rho_old.get_value(q);
          auto        u_old_D          = phi_u_old.get_value(q) - 2.0*scalar_product(phi_u_old.get_value(q), n_plus)*n_plus;

          const auto& avg_flux = 0.5*(phi_u_old.get_value(q) + u_old_D);
          const auto& pres_old     = (EquationData::Cp_Cv - 1.0)*
                                     (phi_E_old.get_value(q) -
                                      0.5/phi_rho_old.get_value(q)*
                                      scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)));
          const auto& pres_old_D   = (EquationData::Cp_Cv - 1.0)*
                                     (phi_E_old.get_value(q) - 0.5/rho_old_D*scalar_product(u_old_D, u_old_D));
          const auto  lambda   = std::max(std::sqrt(scalar_product(phi_u_old.get_value(q)/phi_rho_old.get_value(q),
                                                                   phi_u_old.get_value(q)/phi_rho_old.get_value(q))) +
                                          1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old/phi_rho_old.get_value(q)),
                                          std::sqrt(scalar_product(u_old_D/rho_old_D, u_old_D/rho_old_D)) +
                                          1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_rho_old = phi_rho_old.get_value(q) - rho_old_D;

          phi.submit_value(-a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho_old), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi(data, true, 2),
                                                                         phi_rho_old(data, true, 2),
                                                                         phi_rho_tmp_2(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0), phi_u_tmp_2(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old(data, true, 1), phi_E_tmp_2(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(face);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_E_tmp_2.reinit(face);
        phi_E_tmp_2.gather_evaluate(src[5], true, false);
        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          auto        rho_old_D        = phi_rho_old.get_value(q);
          auto        u_old_D          = phi_u_old.get_value(q) - 2.0*scalar_product(phi_u_old.get_value(q), n_plus)*n_plus;
          auto        rho_tmp_2_D      = phi_rho_tmp_2.get_value(q);
          auto        u_tmp_2_D        = phi_u_tmp_2.get_value(q) - 2.0*scalar_product(phi_u_tmp_2.get_value(q), n_plus)*n_plus;

          const auto& avg_flux_old   = 0.5*(phi_u_old.get_value(q) + u_old_D);
          const auto& pres_old     = (EquationData::Cp_Cv - 1.0)*
                                     (phi_E_old.get_value(q) -
                                      0.5/phi_rho_old.get_value(q)*
                                      scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)));
          const auto& pres_old_D   = (EquationData::Cp_Cv - 1.0)*
                                     (phi_E_old.get_value(q) - 0.5/rho_old_D*scalar_product(u_old_D, u_old_D));
          const auto  lambda_old     = std::max(std::sqrt(scalar_product(phi_u_old.get_value(q)/phi_rho_old.get_value(q),
                                                                         phi_u_old.get_value(q)/phi_rho_old.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old/phi_rho_old.get_value(q)),
                                                std::sqrt(scalar_product(u_old_D/rho_old_D, u_old_D/rho_old_D)) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_rho_old   = phi_rho_old.get_value(q) - rho_old_D;

          const auto& avg_flux_tmp_2 = 0.5*(phi_u_tmp_2.get_value(q) + u_tmp_2_D);
          const auto& pres_tmp_2     = (EquationData::Cp_Cv - 1.0)*
                                       (phi_E_tmp_2.get_value(q) -
                                      0.5/phi_rho_tmp_2.get_value(q)*
                                      scalar_product(phi_u_tmp_2.get_value(q), phi_u_tmp_2.get_value(q)));
          const auto& pres_tmp_2_D   = (EquationData::Cp_Cv - 1.0)*
                                       (phi_E_tmp_2.get_value(q) - 0.5/rho_tmp_2_D*scalar_product(u_tmp_2_D, u_tmp_2_D));
          const auto  lambda_tmp_2   = std::max(std::sqrt(scalar_product(phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q),
                                                                         phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2/phi_rho_tmp_2.get_value(q)),
                                                std::sqrt(scalar_product(u_tmp_2_D/rho_tmp_2_D, u_tmp_2_D/rho_tmp_2_D)) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));
          const auto& jump_rho_tmp_2 = phi_rho_tmp_2.get_value(q) - rho_tmp_2_D;

          phi.submit_value(-a31*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old)
                           -a32*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_rho_projection(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_rho_projection,
                     &HYPERBOLICOperator::assemble_rhs_face_term_rho_projection,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_rho_projection,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_rho_projection(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const Vec&                                   src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi(data, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    Tensor<1, dim, VectorizedArray<Number>> e_k;
    for(unsigned int d = 0; d < dim - 1; ++d)
      e_k[d] = make_vectorized_array<Number>(0.0);
    e_k[dim - 1] = make_vectorized_array<Number>(1.0);

    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0), phi_u_old(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(cell);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& pres_old          = (EquationData::Cp_Cv - 1.0)*
                                          (phi_E_old.get_value(q) -
                                           0.5/phi_rho_old.get_value(q)*
                                           scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)));
          const auto& tensor_product_u_n = outer_product(phi_u_old.get_value(q), phi_u_old.get_value(q)/phi_rho_old.get_value(q));
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = pres_old;

          phi.submit_value(phi_u_old.get_value(q) - a21*dt/(Fr*Fr)*phi_rho_old.get_value(q)*e_k, q);
          phi.submit_gradient(a21*dt*tensor_product_u_n + a21*dt/(Ma*Ma)*p_n_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0), phi_u_old(data, 0), phi_u_tmp_2(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old(data, 1), phi_E_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, 2), phi_rho_tmp_2(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(cell);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_E_tmp_2.reinit(cell);
        phi_E_tmp_2.gather_evaluate(src[5], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& pres_old           = (EquationData::Cp_Cv - 1.0)*
                                           (phi_E_old.get_value(q) -
                                            0.5/phi_rho_old.get_value(q)*
                                            scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)));
          const auto& tensor_product_u_n = outer_product(phi_u_old.get_value(q), phi_u_old.get_value(q)/phi_rho_old.get_value(q));
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = pres_old;

          const auto& pres_tmp_2             = (EquationData::Cp_Cv - 1.0)*
                                               (phi_E_tmp_2.get_value(q) -
                                               0.5/phi_rho_tmp_2.get_value(q)*
                                               scalar_product(phi_u_tmp_2.get_value(q), phi_u_tmp_2.get_value(q)));
          const auto& tensor_product_u_tmp_2 = outer_product(phi_u_tmp_2.get_value(q),
                                                             phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q));
          auto p_tmp_2_times_identity        = tensor_product_u_tmp_2;
          p_tmp_2_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_tmp_2_times_identity[d][d] = pres_tmp_2;

          phi.submit_value(phi_u_old.get_value(q) -
                           a31*dt/(Fr*Fr)*phi_rho_old.get_value(q)*e_k -
                           a32*dt/(Fr*Fr)*phi_rho_tmp_2.get_value(q)*e_k, q);
          phi.submit_gradient(a31*dt*tensor_product_u_n +
                              a31*dt/(Ma*Ma)*p_n_times_identity +
                              a32*dt*tensor_product_u_tmp_2 +
                              a32*dt/(Ma*Ma)*p_tmp_2_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old_p(data, true, 2), phi_rho_old_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                       phi_u_old_p(data, true, 0), phi_u_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old_p(data, true, 1), phi_E_old_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_E_old_p.reinit(face);
        phi_E_old_p.gather_evaluate(src[2], true, false);
        phi_E_old_m.reinit(face);
        phi_E_old_m.gather_evaluate(src[2], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus   = phi_p.get_normal_vector(q);

          const auto& avg_tensor_product_u_n = 0.5*(outer_product(phi_u_old_p.get_value(q),
                                                                  phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q)) +
                                                    outer_product(phi_u_old_m.get_value(q),
                                                                  phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q)));
          const auto& pres_old_p             = (EquationData::Cp_Cv - 1.0)*
                                               (phi_E_old_p.get_value(q) -
                                                0.5/phi_rho_old_p.get_value(q)*
                                                scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q)));
          const auto& pres_old_m             = (EquationData::Cp_Cv - 1.0)*
                                               (phi_E_old_m.get_value(q) -
                                                0.5/phi_rho_old_m.get_value(q)*
                                                scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q)));
          const auto& avg_pres_old = 0.5*(pres_old_p + pres_old_m);
          const auto  lambda = std::max(std::sqrt(scalar_product(phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q),
                                                                 phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q))) +
                                        1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_p/phi_rho_old_p.get_value(q)),
                                        std::sqrt(scalar_product(phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q),
                                                                 phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q))) +
                                        1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_m/phi_rho_old_m.get_value(q)));
          const auto& jump_u_old = phi_u_old_p.get_value(q) - phi_u_old_m.get_value(q);

          phi_p.submit_value(-a21*dt*(avg_tensor_product_u_n*n_plus +
                                      1.0/(Ma*Ma)*avg_pres_old*n_plus +
                                      0.5*lambda*jump_u_old), q);
          phi_m.submit_value(a21*dt*(avg_tensor_product_u_n*n_plus +
                                     1.0/(Ma*Ma)*avg_pres_old*n_plus +
                                     0.5*lambda*jump_u_old), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old_p(data, true, 2), phi_rho_old_m(data, false, 2),
                                                                         phi_rho_tmp_2_p(data, true, 2), phi_rho_tmp_2_m(data, false ,2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0), phi_m(data, false, 0),
                                                                       phi_u_old_p(data, true, 0), phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0), phi_u_tmp_2_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old_p(data, true, 1), phi_E_old_m(data, false, 1),
                                                                     phi_E_tmp_2_p(data, true, 1), phi_E_tmp_2_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_E_old_p.reinit(face);
        phi_E_old_p.gather_evaluate(src[2], true, false);
        phi_E_old_m.reinit(face);
        phi_E_old_m.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_E_tmp_2_p.reinit(face);
        phi_E_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_E_tmp_2_m.reinit(face);
        phi_E_tmp_2_m.gather_evaluate(src[5], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus   = phi_p.get_normal_vector(q);

          const auto& avg_tensor_product_u_n = 0.5*(outer_product(phi_u_old_p.get_value(q),
                                                                  phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q)) +
                                                    outer_product(phi_u_old_m.get_value(q),
                                                                  phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q)));
          const auto& pres_old_p             = (EquationData::Cp_Cv - 1.0)*
                                               (phi_E_old_p.get_value(q) -
                                                0.5/phi_rho_old_p.get_value(q)*
                                                scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q)));
          const auto& pres_old_m             = (EquationData::Cp_Cv - 1.0)*
                                               (phi_E_old_m.get_value(q) -
                                                0.5/phi_rho_old_m.get_value(q)*
                                                scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q)));
          const auto& avg_pres_old           = 0.5*(pres_old_p + pres_old_m);
          const auto  lambda_old  = std::max(std::sqrt(scalar_product(phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q),
                                                                      phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q))) +
                                             1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_p/phi_rho_old_p.get_value(q)),
                                             std::sqrt(scalar_product(phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q),
                                                                      phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q))) +
                                             1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_m/phi_rho_old_m.get_value(q)));
          const auto& jump_u_old = phi_u_old_p.get_value(q) - phi_u_old_m.get_value(q);

          const auto& avg_tensor_product_u_tmp_2 = 0.5*(outer_product(phi_u_tmp_2_p.get_value(q),
                                                                      phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q)) +
                                                        outer_product(phi_u_tmp_2_m.get_value(q),
                                                                      phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q)));
          const auto& pres_tmp_2_p               = (EquationData::Cp_Cv - 1.0)*
                                                   (phi_E_tmp_2_p.get_value(q) -
                                                    0.5/phi_rho_tmp_2_p.get_value(q)*
                                                    scalar_product(phi_u_tmp_2_p.get_value(q), phi_u_tmp_2_p.get_value(q)));
          const auto& pres_tmp_2_m               = (EquationData::Cp_Cv - 1.0)*
                                                   (phi_E_tmp_2_m.get_value(q) -
                                                    0.5/phi_rho_tmp_2_m.get_value(q)*
                                                    scalar_product(phi_u_tmp_2_m.get_value(q), phi_u_tmp_2_m.get_value(q)));
          const auto& avg_pres_tmp_2             = 0.5*(pres_tmp_2_p + pres_tmp_2_m);
          const auto  lambda_tmp_2 = std::max(std::sqrt(scalar_product(phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q),
                                                                         phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q))) +
                                              1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_p/phi_rho_tmp_2_p.get_value(q)),
                                              std::sqrt(scalar_product(phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q),
                                                                       phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q))) +
                                              1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_m/phi_rho_tmp_2_m.get_value(q)));
          const auto& jump_u_tmp_2 = phi_u_tmp_2_p.get_value(q) - phi_u_tmp_2_m.get_value(q);

          phi_p.submit_value(-a31*dt*(avg_tensor_product_u_n*n_plus +
                                      1.0/(Ma*Ma)*avg_pres_old*n_plus +
                                      0.5*lambda_old*jump_u_old)
                             -a32*dt*(avg_tensor_product_u_tmp_2*n_plus +
                                      1.0/(Ma*Ma)*avg_pres_tmp_2*n_plus +
                                      0.5*lambda_tmp_2*jump_u_tmp_2), q);
          phi_m.submit_value(a31*dt*(avg_tensor_product_u_n*n_plus +
                                     1.0/(Ma*Ma)*avg_pres_old*n_plus +
                                     0.5*lambda_old*jump_u_old) +
                             a32*dt*(avg_tensor_product_u_tmp_2*n_plus +
                                     1.0/(Ma*Ma)*avg_pres_tmp_2*n_plus +
                                     0.5*lambda_tmp_2*jump_u_tmp_2), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0), phi_u_old(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(face);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          auto        rho_old_D        = phi_rho_old.get_value(q);
          auto        u_old_D          = phi_u_old.get_value(q) - 2.0*scalar_product(phi_u_old.get_value(q), n_plus)*n_plus;
          auto        pres_old_D       = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old.get_value(q) - 0.5/rho_old_D*scalar_product(u_old_D, u_old_D));

          const auto& avg_tensor_product_u_n = 0.5*(outer_product(phi_u_old.get_value(q),
                                                                  phi_u_old.get_value(q)/phi_rho_old.get_value(q)) +
                                                    outer_product(u_old_D, u_old_D/rho_old_D));
          const auto& pres_old               = (EquationData::Cp_Cv - 1.0)*
                                               (phi_E_old.get_value(q) -
                                                0.5/phi_rho_old.get_value(q)*
                                                scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)));
          const auto& avg_pres_old           = 0.5*(pres_old + pres_old_D);
          const auto  lambda   = std::max(std::sqrt(scalar_product(phi_u_old.get_value(q)/phi_rho_old.get_value(q),
                                                                   phi_u_old.get_value(q)/phi_rho_old.get_value(q))) +
                                          1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old/phi_rho_old.get_value(q)),
                                          std::sqrt(scalar_product(u_old_D/rho_old_D, u_old_D/rho_old_D)) +
                                          1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_u_old = phi_u_old.get_value(q) - u_old_D;

          phi.submit_value(-a21*dt*(avg_tensor_product_u_n*n_plus +
                                    1.0/(Ma*Ma)*avg_pres_old*n_plus +
                                    0.5*lambda*jump_u_old), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, true, 2),
                                                                         phi_rho_tmp_2(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                       phi_u_old(data, true, 0), phi_u_tmp_2(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_E_old(data, true, 1), phi_E_tmp_2(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(face);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_E_tmp_2.reinit(face);
        phi_E_tmp_2.gather_evaluate(src[5], true, false);
        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          auto        rho_old_D        = phi_rho_old.get_value(q);
          auto        u_old_D          = phi_u_old.get_value(q) - 2.0*scalar_product(phi_u_old.get_value(q), n_plus)*n_plus;
          auto        rho_tmp_2_D      = phi_rho_tmp_2.get_value(q);
          auto        u_tmp_2_D        = phi_u_tmp_2.get_value(q) - 2.0*scalar_product(phi_u_tmp_2.get_value(q), n_plus)*n_plus;
          auto        pres_old_D       = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old.get_value(q) - 0.5/rho_old_D*scalar_product(u_old_D, u_old_D));
          auto        pres_tmp_2_D     = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_tmp_2.get_value(q) - 0.5/rho_tmp_2_D*scalar_product(u_tmp_2_D, u_tmp_2_D));

          const auto& avg_tensor_product_u_n = 0.5*(outer_product(phi_u_old.get_value(q),
                                                                  phi_u_old.get_value(q)/phi_rho_old.get_value(q)) +
                                                    outer_product(u_old_D, u_old_D/rho_old_D));
          const auto& pres_old               = (EquationData::Cp_Cv - 1.0)*
                                               (phi_E_old.get_value(q) -
                                                0.5/phi_rho_old.get_value(q)*
                                                scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)));
          const auto& avg_pres_old           = 0.5*(pres_old + pres_old_D);
          const auto  lambda_old     = std::max(std::sqrt(scalar_product(phi_u_old.get_value(q)/phi_rho_old.get_value(q),
                                                                         phi_u_old.get_value(q)/phi_rho_old.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old/phi_rho_old.get_value(q)),
                                                std::sqrt(scalar_product(u_old_D/rho_old_D, u_old_D/rho_old_D)) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_u_old = phi_u_old.get_value(q) - u_old_D;

          const auto& avg_tensor_product_u_tmp_2 = 0.5*(outer_product(phi_u_tmp_2.get_value(q),
                                                                      phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q)) +
                                                        outer_product(u_tmp_2_D, u_tmp_2_D/rho_tmp_2_D));
          const auto& pres_tmp_2                 = (EquationData::Cp_Cv - 1.0)*
                                                   (phi_E_tmp_2.get_value(q) -
                                                    0.5/phi_rho_tmp_2.get_value(q)*
                                                    scalar_product(phi_u_tmp_2.get_value(q), phi_u_tmp_2.get_value(q)));
          const auto& avg_pres_tmp_2             = 0.5*(pres_tmp_2 + pres_tmp_2_D);
          const auto  lambda_tmp_2   = std::max(std::sqrt(scalar_product(phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q),
                                                                         phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2/phi_rho_tmp_2.get_value(q)),
                                                std::sqrt(scalar_product(u_tmp_2_D/rho_tmp_2_D, u_tmp_2_D/rho_tmp_2_D)) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));
          const auto& jump_u_tmp_2 = phi_u_tmp_2.get_value(q) - u_tmp_2_D;

          phi.submit_value(-a31*dt*(avg_tensor_product_u_n*n_plus +
                                    1.0/(Ma*Ma)*avg_pres_old*n_plus +
                                    0.5*lambda_old*jump_u_old)
                           -a32*dt*(avg_tensor_product_u_tmp_2*n_plus +
                                    1.0/(Ma*Ma)*avg_pres_tmp_2*n_plus +
                                    0.5*lambda_tmp_2*jump_u_tmp_2), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_momentum_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_momentum_update,
                     &HYPERBOLICOperator::assemble_rhs_face_term_momentum_update,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_momentum_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the energy update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_energy(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const std::vector<Vec>&                      src,
                                const std::pair<unsigned int, unsigned int>& cell_range) const {
    Tensor<1, dim, VectorizedArray<Number>> e_k;
    for(unsigned int d = 0; d < dim - 1; ++d)
      e_k[d] = make_vectorized_array<Number>(0.0);
    e_k[dim - 1] = make_vectorized_array<Number>(1.0);

    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1), phi_E_old(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(cell);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old  = phi_rho_old.get_value(q);
          const auto& u_old    = phi_u_old.get_value(q);
          const auto& E_old    = phi_E_old.get_value(q);
          const auto& pres_old = (EquationData::Cp_Cv - 1.0)*(E_old - 0.5/rho_old*scalar_product(u_old, u_old));

          phi.submit_value(E_old - a21*dt*Ma*Ma/(Fr*Fr)*u_old[dim - 1], q);
          phi.submit_gradient(a21*dt*(Ma*Ma*0.5*scalar_product(u_old/rho_old, u_old/rho_old)*u_old +
                                      (E_old - 0.5*scalar_product(u_old, u_old)/rho_old + pres_old)*u_old/rho_old), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1), phi_E_old(data, 1), phi_E_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0), phi_u_tmp_2(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, 2), phi_rho_tmp_2(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(cell);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_E_tmp_2.reinit(cell);
        phi_E_tmp_2.gather_evaluate(src[5], true, false);
        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old    = phi_rho_old.get_value(q);
          const auto& u_old      = phi_u_old.get_value(q);
          const auto& E_old      = phi_E_old.get_value(q);
          const auto& pres_old   = (EquationData::Cp_Cv - 1.0)*(E_old - 0.5/rho_old*scalar_product(u_old, u_old));
          const auto& rho_tmp_2  = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2    = phi_u_tmp_2.get_value(q);
          const auto& E_tmp_2    = phi_E_tmp_2.get_value(q);
          const auto& pres_tmp_2 = (EquationData::Cp_Cv - 1.0)*(E_tmp_2 - 0.5/rho_tmp_2*scalar_product(u_tmp_2, u_tmp_2));

          phi.submit_value(E_old - a31*dt*Ma*Ma/(Fr*Fr)*u_old[dim - 1] - a32*dt*Ma*Ma/(Fr*Fr)*u_tmp_2[dim - 1], q);
          phi.submit_gradient(a31*dt*(Ma*Ma*0.5*scalar_product(u_old/rho_old, u_old/rho_old)*u_old +
                                      (E_old - 0.5*scalar_product(u_old, u_old)/rho_old + pres_old)*u_old/rho_old) +
                              a32*dt*(Ma*Ma*0.5*scalar_product(u_tmp_2/rho_tmp_2, u_tmp_2/rho_tmp_2)*u_tmp_2 +
                                      (E_tmp_2 - 0.5*scalar_product(u_tmp_2, u_tmp_2)/rho_tmp_2 + pres_tmp_2)*u_tmp_2/rho_tmp_2), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the energy update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_energy(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const std::vector<Vec>&                      src,
                                const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_p(data, true, 1), phi_m(data, false, 1),
                                                                     phi_E_old_p(data, true, 1), phi_E_old_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old_p(data, true, 2), phi_rho_old_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0), phi_u_old_m(data, false, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_E_old_p.reinit(face);
        phi_E_old_p.gather_evaluate(src[2], true, false);
        phi_E_old_m.reinit(face);
        phi_E_old_m.gather_evaluate(src[2], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus   = phi_p.get_normal_vector(q);

          const auto& pres_old_p       = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old_p.get_value(q) -
                                          0.5/phi_rho_old_p.get_value(q)*
                                          scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q)));
          const auto& pres_old_m       = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old_m.get_value(q) -
                                          0.5/phi_rho_old_m.get_value(q)*
                                          scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q)));
          const auto& avg_enthalpy_old = 0.5*((phi_E_old_p.get_value(q) -
                                               0.5/phi_rho_old_p.get_value(q)*
                                               scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q)) + pres_old_p)*
                                               phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q) +
                                               (phi_E_old_m.get_value(q) -
                                                0.5/phi_rho_old_m.get_value(q)*
                                                scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q)) + pres_old_m)*
                                                phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q));
          const auto& avg_kinetic_old = 0.5*(0.5*phi_u_old_p.get_value(q)*
                                             scalar_product(phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q),
                                                            phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q)) +
                                             0.5*phi_u_old_m.get_value(q)*
                                             scalar_product(phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q),
                                                            phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q)));
          const auto  lambda     = std::max(std::sqrt(scalar_product(phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q),
                                                                     phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q))) +
                                            1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_p/phi_rho_old_p.get_value(q)),
                                            std::sqrt(scalar_product(phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q),
                                                                     phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q))) +
                                            1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_m/phi_rho_old_m.get_value(q)));
          const auto& jump_E_old = phi_E_old_p.get_value(q) - phi_E_old_m.get_value(q);

          phi_p.submit_value(-a21*dt*(scalar_product(avg_enthalpy_old, n_plus) +
                                      Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                                      0.5*lambda*jump_E_old), q);
          phi_m.submit_value(a21*dt*(scalar_product(avg_enthalpy_old, n_plus) +
                                     Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                                     0.5*lambda*jump_E_old), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_p(data, true, 1), phi_m(data, false, 1),
                                                                     phi_E_old_p(data, true, 1), phi_E_old_m(data, false, 1),
                                                                     phi_E_tmp_2_p(data, true, 1), phi_E_tmp_2_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old_p(data, true, 2), phi_rho_old_m(data, false, 2),
                                                                         phi_rho_tmp_2_p(data, true, 2), phi_rho_tmp_2_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0), phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0), phi_u_tmp_2_m(data, false, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_E_old_p.reinit(face);
        phi_E_old_p.gather_evaluate(src[2], true, false);
        phi_E_old_m.reinit(face);
        phi_E_old_m.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_E_tmp_2_p.reinit(face);
        phi_E_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_E_tmp_2_m.reinit(face);
        phi_E_tmp_2_m.gather_evaluate(src[5], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus   = phi_p.get_normal_vector(q);

          const auto& pres_old_p       = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old_p.get_value(q) -
                                          0.5/phi_rho_old_p.get_value(q)*
                                          scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q)));
          const auto& pres_old_m       = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old_m.get_value(q) -
                                          0.5/phi_rho_old_m.get_value(q)*
                                          scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q)));
          const auto& avg_enthalpy_old = 0.5*((phi_E_old_p.get_value(q) -
                                               0.5/phi_rho_old_p.get_value(q)*
                                               scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q)) + pres_old_p)*
                                               phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q) +
                                               (phi_E_old_m.get_value(q) -
                                                0.5/phi_rho_old_m.get_value(q)*
                                                scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q)) + pres_old_m)*
                                                phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q));
          const auto& avg_kinetic_old = 0.5*(0.5*phi_u_old_p.get_value(q)*
                                             scalar_product(phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q),
                                                            phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q)) +
                                             0.5*phi_u_old_m.get_value(q)*
                                             scalar_product(phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q),
                                                            phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q)));
          const auto  lambda_old     = std::max(std::sqrt(scalar_product(phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q),
                                                                         phi_u_old_p.get_value(q)/phi_rho_old_p.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_p/phi_rho_old_p.get_value(q)),
                                                std::sqrt(scalar_product(phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q),
                                                                         phi_u_old_m.get_value(q)/phi_rho_old_m.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_m/phi_rho_old_m.get_value(q)));
          const auto& jump_E_old = phi_E_old_p.get_value(q) - phi_E_old_m.get_value(q);

          const auto& pres_tmp_2_p       = (EquationData::Cp_Cv - 1.0)*
                                           (phi_E_tmp_2_p.get_value(q) -
                                            0.5/phi_rho_tmp_2_p.get_value(q)*
                                            scalar_product(phi_u_tmp_2_p.get_value(q), phi_u_tmp_2_p.get_value(q)));
          const auto& pres_tmp_2_m       = (EquationData::Cp_Cv - 1.0)*
                                           (phi_E_tmp_2_m.get_value(q) -
                                            0.5/phi_rho_tmp_2_m.get_value(q)*
                                            scalar_product(phi_u_tmp_2_m.get_value(q), phi_u_tmp_2_m.get_value(q)));
          const auto& avg_enthalpy_tmp_2 = 0.5*((phi_E_tmp_2_p.get_value(q) -
                                                 0.5/phi_rho_tmp_2_p.get_value(q)*
                                                 scalar_product(phi_u_tmp_2_p.get_value(q), phi_u_tmp_2_p.get_value(q)) +
                                                 pres_tmp_2_p)*
                                                 phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q) +
                                                (phi_E_tmp_2_m.get_value(q) -
                                                 0.5/phi_rho_tmp_2_m.get_value(q)*
                                                 scalar_product(phi_u_tmp_2_m.get_value(q), phi_u_tmp_2_m.get_value(q)) +
                                                 pres_tmp_2_m)*
                                                 phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q));
          const auto& avg_kinetic_tmp_2 = 0.5*(0.5*phi_u_tmp_2_p.get_value(q)*
                                               scalar_product(phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q),
                                                              phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q)) +
                                               0.5*phi_u_tmp_2_m.get_value(q)*
                                               scalar_product(phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q),
                                                              phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q)));
          const auto  lambda_tmp_2 = std::max(std::sqrt(scalar_product(phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q),
                                                                         phi_u_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q))) +
                                              1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_p/phi_rho_tmp_2_p.get_value(q)),
                                              std::sqrt(scalar_product(phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q),
                                                                       phi_u_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q))) +
                                              1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_m/phi_rho_tmp_2_m.get_value(q)));
          const auto& jump_E_tmp_2 = phi_E_tmp_2_p.get_value(q) - phi_E_tmp_2_m.get_value(q);

          phi_p.submit_value(-a31*dt*(scalar_product(avg_enthalpy_old, n_plus) +
                                      Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                                      0.5*lambda_old*jump_E_old)
                             -a32*dt*(scalar_product(avg_enthalpy_tmp_2, n_plus) +
                                      Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus) +
                                      0.5*lambda_tmp_2*jump_E_tmp_2), q);
          phi_m.submit_value(a31*dt*(scalar_product(avg_enthalpy_old, n_plus) +
                                     Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                                     0.5*lambda_old*jump_E_old) +
                             a32*dt*(scalar_product(avg_enthalpy_tmp_2, n_plus) +
                                     Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus) +
                                     0.5*lambda_tmp_2*jump_E_tmp_2), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the energy update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_energy(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const std::vector<Vec>&                      src,
                                    const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, true, 1),
                                                                     phi_E_old(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(face);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          auto        rho_old_D        = phi_rho_old.get_value(q);
          auto        u_old_D          = phi_u_old.get_value(q) - 2.0*scalar_product(phi_u_old.get_value(q), n_plus)*n_plus;
          auto        E_old_D          = phi_E_old.get_value(q);
          auto        pres_old_D       = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old.get_value(q) - 0.5/rho_old_D*scalar_product(u_old_D, u_old_D));

          const auto& pres_old         = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old.get_value(q) -
                                          0.5/phi_rho_old.get_value(q)*
                                          scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)));
          const auto& avg_enthalpy_old = 0.5*((phi_E_old.get_value(q) -
                                               0.5/phi_rho_old.get_value(q)*
                                               scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)) + pres_old)*
                                               phi_u_old.get_value(q)/phi_rho_old.get_value(q)  +
                                               (E_old_D - 0.5/rho_old_D*scalar_product(u_old_D, u_old_D) + pres_old_D)*
                                               u_old_D/rho_old_D);
          const auto& avg_kinetic_old = 0.5*(0.5*scalar_product(phi_u_old.get_value(q)/phi_rho_old.get_value(q),
                                                                phi_u_old.get_value(q)/phi_rho_old.get_value(q))*
                                             phi_u_old.get_value(q) +
                                             0.5*scalar_product(u_old_D/rho_old_D, u_old_D/rho_old_D)*u_old_D);
          const auto  lambda   = std::max(std::sqrt(scalar_product(phi_u_old.get_value(q)/phi_rho_old.get_value(q),
                                                                   phi_u_old.get_value(q)/phi_rho_old.get_value(q))) +
                                          1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old/phi_rho_old.get_value(q)),
                                          std::sqrt(scalar_product(u_old_D/rho_old_D, u_old_D/rho_old_D)) +
                                          1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_E_old = phi_E_old.get_value(q) - E_old_D;

          phi.submit_value(-a21*dt*(scalar_product(avg_enthalpy_old, n_plus) +
                                    Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                                    0.5*lambda*jump_E_old), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, true, 1),
                                                                     phi_E_old(data, true, 1),
                                                                     phi_E_tmp_2(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, true, 2), phi_rho_tmp_2(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0), phi_u_tmp_2(data, true, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_E_old.reinit(face);
        phi_E_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_E_tmp_2.reinit(face);
        phi_E_tmp_2.gather_evaluate(src[5], true, false);
        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          auto        rho_old_D        = phi_rho_old.get_value(q);
          auto        E_old_D          = phi_E_old.get_value(q);
          auto        u_old_D          = phi_u_old.get_value(q) - 2.0*scalar_product(phi_u_old.get_value(q), n_plus)*n_plus;
          auto        rho_tmp_2_D      = phi_rho_tmp_2.get_value(q);
          auto        E_tmp_2_D        = phi_E_tmp_2.get_value(q);
          auto        u_tmp_2_D        = phi_u_tmp_2.get_value(q) - 2.0*scalar_product(phi_u_tmp_2.get_value(q), n_plus)*n_plus;
          auto        pres_old_D       = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old.get_value(q) - 0.5/rho_old_D*scalar_product(u_old_D, u_old_D));
          auto        pres_tmp_2_D     = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_tmp_2.get_value(q) - 0.5/rho_tmp_2_D*scalar_product(u_tmp_2_D, u_tmp_2_D));

          const auto& pres_old         = (EquationData::Cp_Cv - 1.0)*
                                         (phi_E_old.get_value(q) -
                                          0.5/phi_rho_old.get_value(q)*
                                          scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)));
          const auto& avg_enthalpy_old = 0.5*((phi_E_old.get_value(q) -
                                               0.5/phi_rho_old.get_value(q)*
                                               scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q)) + pres_old)*
                                               phi_u_old.get_value(q)/phi_rho_old.get_value(q)  +
                                               (E_old_D - 0.5/rho_old_D*scalar_product(u_old_D, u_old_D) + pres_old_D)*
                                               u_old_D/rho_old_D);
          const auto& avg_kinetic_old = 0.5*(0.5*scalar_product(phi_u_old.get_value(q)/phi_rho_old.get_value(q),
                                                                phi_u_old.get_value(q)/phi_rho_old.get_value(q))*
                                             phi_u_old.get_value(q) +
                                             0.5*scalar_product(u_old_D/rho_old_D, u_old_D/rho_old_D)*u_old_D);
          const auto  lambda_old     = std::max(std::sqrt(scalar_product(phi_u_old.get_value(q)/phi_rho_old.get_value(q),
                                                                         phi_u_old.get_value(q)/phi_rho_old.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old/phi_rho_old.get_value(q)),
                                                std::sqrt(scalar_product(u_old_D/rho_old_D, u_old_D/rho_old_D)) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_E_old = phi_E_old.get_value(q) - E_old_D;

          const auto& pres_tmp_2         = (EquationData::Cp_Cv - 1.0)*
                                           (phi_E_tmp_2.get_value(q) -
                                            0.5/phi_rho_tmp_2.get_value(q)*
                                            scalar_product(phi_u_tmp_2.get_value(q), phi_u_tmp_2.get_value(q)));
          const auto& avg_enthalpy_tmp_2 = 0.5*((phi_E_tmp_2.get_value(q) -
                                                 0.5/phi_rho_tmp_2.get_value(q)*
                                                 scalar_product(phi_u_tmp_2.get_value(q), phi_u_tmp_2.get_value(q)) + pres_tmp_2)*
                                                 phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q)  +
                                                 (E_tmp_2_D - 0.5/rho_tmp_2_D*scalar_product(u_tmp_2_D, u_tmp_2_D) + pres_tmp_2_D)*
                                                 u_tmp_2_D/rho_tmp_2_D);
          const auto& avg_kinetic_tmp_2 = 0.5*(0.5*scalar_product(phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q),
                                                                  phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q))*
                                               phi_u_tmp_2.get_value(q) +
                                               0.5*scalar_product(u_tmp_2_D/rho_tmp_2_D, u_tmp_2_D/rho_tmp_2_D)*u_tmp_2_D);
          const auto  lambda_tmp_2   = std::max(std::sqrt(scalar_product(phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q),
                                                                         phi_u_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q))) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2/phi_rho_tmp_2.get_value(q)),
                                                std::sqrt(scalar_product(u_tmp_2_D/rho_tmp_2_D, u_tmp_2_D/rho_tmp_2_D)) +
                                                1.0/Ma*std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));
          const auto& jump_E_tmp_2 = phi_E_tmp_2.get_value(q) - E_tmp_2_D;

          phi.submit_value(-a31*dt*(scalar_product(avg_enthalpy_old, n_plus) +
                                    Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                                    0.5*lambda_old*jump_E_old)
                           -a32*dt*(scalar_product(avg_enthalpy_tmp_2, n_plus) +
                                    Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus) +
                                    0.5*lambda_tmp_2*jump_E_tmp_2), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_energy(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_energy,
                     &HYPERBOLICOperator::assemble_rhs_face_term_energy,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_energy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_energy(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const Vec&                                   src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the velocity projection
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_velocity_projection(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0), phi_u_curr(data, 0);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_curr(data, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_curr.reinit(cell);
      phi_rho_curr.gather_evaluate(src[0], true, false);
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(src[1], true, true);
      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi_u_curr.get_value(q)/phi_rho_curr.get_value(q), q);

      phi.integrate_scatter(true, false, dst);
    }
  }


  // Put together all the previous steps for velocity projection
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_velocity_projection(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->cell_loop(&HYPERBOLICOperator::assemble_rhs_cell_term_velocity_projection,
                          this, dst, src, true);
  }


  // Assemble cell term for the velocity projection
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_velocity_projection(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the pressure projection
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_pressure_projection(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1), phi_E(data, 1);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho(data, 2);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u(data, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho.reinit(cell);
      phi_rho.gather_evaluate(src[0], true, false);
      phi_u.reinit(cell);
      phi_u.gather_evaluate(src[1], true, true);
      phi_E.reinit(cell);
      phi_E.gather_evaluate(src[2], true, false);
      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value((EquationData::Cp_Cv - 1.0)*
                         (phi_E.get_value(q) -
                          0.5/phi_rho.get_value(q)*scalar_product(phi_u.get_value(q), phi_u.get_value(q))), q);

      phi.integrate_scatter(true, false, dst);
    }
  }


  // Put together all the previous steps for pressure projection
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_pressure_projection(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->cell_loop(&HYPERBOLICOperator::assemble_rhs_cell_term_pressure_projection,
                          this, dst, src, true);
  }


  // Assemble cell term for the pressure projection
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_pressure_projection(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(NS_stage, 7);
    Assert(NS_stage > 0, ExcInternalError());
    if(NS_stage == 1) {
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_rho_projection,
                            this, dst, src, false);
    }
    else if(NS_stage == 2){
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_momentum_update,
                            this, dst, src, false);
    }
    else if(NS_stage == 3){
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_energy,
                            this, dst, src, false);
    }
    else if(NS_stage == 4){
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_velocity_projection,
                            this, dst, src, false);
    }
    else if(NS_stage == 5){
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_pressure_projection,
                            this, dst, src, false);
    }
  }


  // @sect3{The <code>NavierStokesProjection</code> class}

  // Now for the main class of the program. It implements the various versions
  // of the projection method for Navier-Stokes equations.
  //
  template<int dim>
  class NavierStokesProjection {
  public:
    NavierStokesProjection(RunTimeParameters::Data_Storage& data);

    void run(const bool verbose = false, const unsigned int output_interval = 10);

  protected:
    const double t_0;
    const double T;
    const double gamma;            //--- TR-BDF2 parameter
    unsigned int HYPERBOLIC_stage; //--- Flag to check at which current stage of TR-BDF2 are
    const double Ma;
    const double Fr;
    double       dt;
    const double CFL;

    parallel::distributed::Triangulation<dim> triangulation;

    FESystem<dim> fe_density;
    FESystem<dim> fe_velocity;
    FESystem<dim> fe_temperature;

    DoFHandler<dim> dof_handler_density;
    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_temperature;

    QGauss<dim> quadrature_density;
    QGauss<dim> quadrature_velocity;
    QGauss<dim> quadrature_temperature;

    LinearAlgebra::distributed::Vector<double> rho_old;
    LinearAlgebra::distributed::Vector<double> rho_tmp_2;
    LinearAlgebra::distributed::Vector<double> rho_curr;
    LinearAlgebra::distributed::Vector<double> rhs_rho;

    LinearAlgebra::distributed::Vector<double> u_old;
    LinearAlgebra::distributed::Vector<double> u_tmp_2;
    LinearAlgebra::distributed::Vector<double> u_curr;
    LinearAlgebra::distributed::Vector<double> rhs_u;
    LinearAlgebra::distributed::Vector<double> u_proj;

    LinearAlgebra::distributed::Vector<double> E_old;
    LinearAlgebra::distributed::Vector<double> E_tmp_2;
    LinearAlgebra::distributed::Vector<double> E_curr;
    LinearAlgebra::distributed::Vector<double> rhs_E;
    LinearAlgebra::distributed::Vector<double> pres_old;
    LinearAlgebra::distributed::Vector<double> pres_tmp_2;
    LinearAlgebra::distributed::Vector<double> rhs_pres;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation(const unsigned int n_refines);

    void setup_dofs();

    void initialize();

    void update_density();

    void update_momentum();

    void update_energy();

    void output_results(const unsigned int step);

  private:
    EquationData::Density<dim>  rho_init;
    EquationData::Velocity<dim> u_init;
    EquationData::Energy<dim>   E_init;

    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    HYPERBOLICOperator<dim, EquationData::degree_rho, EquationData::degree_T, EquationData::degree_u,
                            2*EquationData::degree_rho + 1, 2*EquationData::degree_T + 1, 2*EquationData::degree_u + 1,
                            LinearAlgebra::distributed::Vector<double>, double> euler_matrix;

    MGLevelObject<HYPERBOLICOperator<dim, EquationData::degree_rho, EquationData::degree_T, EquationData::degree_u,
                                          2*EquationData::degree_rho + 1, 2*EquationData::degree_T + 1, 2*EquationData::degree_u + 1,
                                          LinearAlgebra::distributed::Vector<float>, float>> mg_matrices;

    AffineConstraints<double> constraints_velocity,
                              constraints_temperature,
                              constraints_density;

    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    std::string saving_dir;

    ConditionalOStream pcout;

    std::ofstream      time_out;
    ConditionalOStream ptime_out;
    TimerOutput        time_table;

    std::ofstream output_n_dofs_velocity;
    std::ofstream output_n_dofs_temperature;
    std::ofstream output_n_dofs_density;

    std::ofstream output_error_vel;
    std::ofstream output_error_rho;
    std::ofstream output_error_pres;

    Vector<double> H1_error_per_cell_vel,
                   L2_error_per_cell_vel,
                   H1_rel_error_per_cell_vel,
                   L2_rel_error_per_cell_vel,
                   H1_error_per_cell_rho,
                   L2_error_per_cell_rho,
                   H1_rel_error_per_cell_rho,
                   L2_rel_error_per_cell_rho,
                   H1_error_per_cell_energy,
                   L2_error_per_cell_energy,
                   H1_rel_error_per_cell_energy,
                   L2_rel_error_per_cell_energy,
                   Linfty_error_per_cell_vel;

    void project_velocity();

    void project_pressure();

    double get_maximal_velocity();
  };


  // @sect4{ <code>NavierStokesProjection::NavierStokesProjection</code> }

  // In the constructor, we just read all the data from the
  // <code>Data_Storage</code> object that is passed as an argument, verify that
  // the data we read are reasonable and, finally, create the triangulation and
  // load the initial data.
  template<int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(RunTimeParameters::Data_Storage& data):
    t_0(data.initial_time),
    T(data.final_time),
    gamma(2.0 - std::sqrt(2.0)),     //--- Save also in the NavierStokes class the TR-BDF2 parameter value
    HYPERBOLIC_stage(1),             //--- Initialize the flag for the TR_BDF2 stage
    Ma(data.Mach),
    Fr(data.Froude),
    dt(data.dt),
    CFL(data.CFL),
    triangulation(MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    fe_density(FE_DGQ<dim>(EquationData::degree_rho), 1),
    fe_velocity(FE_DGQ<dim>(EquationData::degree_u), dim),
    fe_temperature(FE_DGQ<dim>(EquationData::degree_T), 1),
    dof_handler_density(triangulation),
    dof_handler_velocity(triangulation),
    dof_handler_temperature(triangulation),
    quadrature_density(2*EquationData::degree_rho + 1),
    quadrature_velocity(2*EquationData::degree_u + 1),
    quadrature_temperature(2*EquationData::degree_T + 1),
    rho_init(data.initial_time),
    u_init(data.initial_time),
    E_init(data.initial_time),
    euler_matrix(data),
    vel_max_its(data.vel_max_iterations),
    vel_Krylov_size(data.vel_Krylov_size),
    vel_off_diagonals(data.vel_off_diagonals),
    vel_update_prec(data.vel_update_prec),
    vel_eps(data.vel_eps),
    vel_diag_strength(data.vel_diag_strength),
    saving_dir(data.dir),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_" +
             Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    output_n_dofs_velocity("./" + data.dir + "/n_dofs_velocity.dat", std::ofstream::out),
    output_n_dofs_temperature("./" + data.dir + "/n_dofs_temperature.dat", std::ofstream::out),
    output_n_dofs_density("./" + data.dir + "/n_dofs_density.dat", std::ofstream::out),
    output_error_vel("./" + data.dir + "/error_analysis_vel.dat", std::ofstream::out),
    output_error_rho("./" + data.dir + "/error_analysis_rho.dat", std::ofstream::out),
    output_error_pres("./" + data.dir + "/error_analysis_pres.dat", std::ofstream::out) {
      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

      constraints_velocity.clear();
      constraints_temperature.clear();
      constraints_density.clear();

      create_triangulation(data.n_global_refines);
      setup_dofs();
      initialize();
  }


  // @sect4{<code>NavierStokesProjection::create_triangulation_and_dofs</code>}

  // The method that creates the triangulation.
  //
  template<int dim>
  void NavierStokesProjection<dim>::create_triangulation(const unsigned int n_refines) {
    TimerOutput::Scope t(time_table, "Create triangulation");

    Point<dim> lower_left;
    Point<dim> upper_right;
    upper_right[0] = 1000.0;
    upper_right[1] = 1000.0;

    GridGenerator::hyper_rectangle(triangulation, lower_left, upper_right, true);
    triangulation.refine_global(2);
    triangulation.refine_global(n_refines);
  }


  // After creating the triangulation, it creates the mesh dependent
  // data, i.e. it distributes degrees of freedom and renumbers them, and
  // initializes the matrices and vectors that we will use.
  //
  template<int dim>
  void NavierStokesProjection<dim>::setup_dofs() {
    pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
    pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_temperature.distribute_dofs(fe_temperature);
    dof_handler_density.distribute_dofs(fe_density);

    mg_matrices.clear_elements();
    dof_handler_velocity.distribute_mg_dofs();
    dof_handler_temperature.distribute_mg_dofs();
    dof_handler_density.distribute_mg_dofs();

    pcout << "dim (V_h) = " << dof_handler_velocity.n_dofs()
          << std::endl
          << "dim (Q_h) = " << dof_handler_temperature.n_dofs()
          << std::endl
          << "dim (X_h) = " << dof_handler_density.n_dofs()
          << std::endl
          << "Ma        = " << Ma
          << std::endl
          << "Fr        = " << Fr << std::endl
          << std::endl;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_n_dofs_velocity    << dof_handler_velocity.n_dofs()    << std::endl;
      output_n_dofs_temperature << dof_handler_temperature.n_dofs() << std::endl;
      output_n_dofs_density     << dof_handler_density.n_dofs()     << std::endl;
    }

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points | update_values);
    additional_data.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                        update_normal_vectors | update_values);
    additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                           update_normal_vectors | update_values);
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;

    std::vector<const DoFHandler<dim>*> dof_handlers;
    dof_handlers.push_back(&dof_handler_velocity);
    dof_handlers.push_back(&dof_handler_temperature);
    dof_handlers.push_back(&dof_handler_density);

    std::vector<const AffineConstraints<double>*> constraints;
    constraints.push_back(&constraints_velocity);
    constraints.push_back(&constraints_temperature);
    constraints.push_back(&constraints_density);

    std::vector<QGauss<1>> quadratures;
    quadratures.push_back(QGauss<1>(2*EquationData::degree_u + 1));
    quadratures.push_back(QGauss<1>(2*EquationData::degree_T + 1));
    quadratures.push_back(QGauss<1>(2*EquationData::degree_rho + 1));

    matrix_free_storage->reinit(dof_handlers, constraints, quadratures, additional_data);

    matrix_free_storage->initialize_dof_vector(u_old, 0);
    matrix_free_storage->initialize_dof_vector(u_tmp_2, 0);
    matrix_free_storage->initialize_dof_vector(u_curr, 0);
    matrix_free_storage->initialize_dof_vector(rhs_u, 0);
    matrix_free_storage->initialize_dof_vector(u_proj, 0);

    matrix_free_storage->initialize_dof_vector(E_old, 1);
    matrix_free_storage->initialize_dof_vector(E_tmp_2, 1);
    matrix_free_storage->initialize_dof_vector(E_curr, 1);
    matrix_free_storage->initialize_dof_vector(rhs_E, 1);
    matrix_free_storage->initialize_dof_vector(pres_old, 1);
    matrix_free_storage->initialize_dof_vector(pres_tmp_2, 1);
    matrix_free_storage->initialize_dof_vector(rhs_pres, 1);

    matrix_free_storage->initialize_dof_vector(rho_old, 2);
    matrix_free_storage->initialize_dof_vector(rho_tmp_2, 2);
    matrix_free_storage->initialize_dof_vector(rho_curr, 2);
    matrix_free_storage->initialize_dof_vector(rhs_rho, 2);

    Vector<double> error_per_cell_tmp(triangulation.n_active_cells());
    H1_error_per_cell_vel.reinit(error_per_cell_tmp);
    L2_error_per_cell_vel.reinit(error_per_cell_tmp);
    H1_rel_error_per_cell_vel.reinit(error_per_cell_tmp);
    L2_rel_error_per_cell_vel.reinit(error_per_cell_tmp);
    H1_error_per_cell_rho.reinit(error_per_cell_tmp);
    L2_error_per_cell_rho.reinit(error_per_cell_tmp);
    H1_rel_error_per_cell_rho.reinit(error_per_cell_tmp);
    L2_rel_error_per_cell_rho.reinit(error_per_cell_tmp);
    H1_error_per_cell_energy.reinit(error_per_cell_tmp);
    L2_error_per_cell_energy.reinit(error_per_cell_tmp);
    H1_rel_error_per_cell_energy.reinit(error_per_cell_tmp);
    L2_rel_error_per_cell_energy.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_vel.reinit(error_per_cell_tmp);
  }


  // @sect4{ <code>NavierStokesProjection::initialize</code> }

  // This method loads the initial data
  //
  template<int dim>
  void NavierStokesProjection<dim>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize state");

    VectorTools::interpolate(dof_handler_density, rho_init, rho_old);
    VectorTools::interpolate(dof_handler_velocity, u_init, u_old);
    VectorTools::interpolate(dof_handler_temperature, E_init, E_old);
  }


  // @sect4{<code>NavierStokesProjection::update_density</code>}

  // This implements the update of the density for the hyperbolic part
  //
  template<int dim>
  void NavierStokesProjection<dim>::update_density() {
    TimerOutput::Scope t(time_table, "Update density");

    const std::vector<unsigned int> tmp = {2};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);

    euler_matrix.set_NS_stage(1);

    if(HYPERBOLIC_stage == 1)
      euler_matrix.vmult_rhs_rho_projection(rhs_rho, {rho_old, u_old, E_old});
    else
      euler_matrix.vmult_rhs_rho_projection(rhs_rho, {rho_old, u_old, E_old,
                                                      rho_tmp_2, u_tmp_2, E_tmp_2});

    SolverControl solver_control(vel_max_its, 1e-12*rhs_rho.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    if(HYPERBOLIC_stage == 1) {
      rho_tmp_2 = rho_old;
      cg.solve(euler_matrix, rho_tmp_2, rhs_rho, PreconditionIdentity());
    }
    else {
      rho_curr = rho_tmp_2;
      cg.solve(euler_matrix, rho_curr, rhs_rho, PreconditionIdentity());
    }
  }


  // @sect4{<code>NavierStokesProjection::update_momentum</code>}

  // This implements the momentum update
  //
  template<int dim>
  void NavierStokesProjection<dim>::update_momentum() {
    TimerOutput::Scope t(time_table, "Update momentum");

    const std::vector<unsigned int> tmp = {0};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);

    euler_matrix.set_NS_stage(2);

    if(HYPERBOLIC_stage == 1)
      euler_matrix.vmult_rhs_momentum_update(rhs_u, {rho_old, u_old, E_old});
    else
      euler_matrix.vmult_rhs_momentum_update(rhs_u, {rho_old, u_old, E_old,
                                                     rho_tmp_2, u_tmp_2, E_tmp_2});

    SolverControl solver_control(vel_max_its, 1e-12*rhs_u.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    if(HYPERBOLIC_stage == 1) {
      u_tmp_2.equ(1.0, u_old);
      cg.solve(euler_matrix, u_tmp_2, rhs_u, PreconditionIdentity());
    }
    else {
      u_curr.equ(1.0, u_tmp_2);
      cg.solve(euler_matrix, u_curr, rhs_u, PreconditionIdentity());
    }
  }


  // @sect4{<code>NavierStokesProjection::update_energy</code>}

  // This implements the update of total energy density
  //
  template<int dim>
  void NavierStokesProjection<dim>::update_energy() {
    TimerOutput::Scope t(time_table, "Update energy");

    const std::vector<unsigned int> tmp = {1};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);

    euler_matrix.set_NS_stage(3);

    if(HYPERBOLIC_stage == 1)
      euler_matrix.vmult_rhs_energy(rhs_E, {rho_old, u_old, E_old});
    else
      euler_matrix.vmult_rhs_energy(rhs_E, {rho_old, u_old, E_old,
                                            rho_tmp_2, u_tmp_2, E_tmp_2});

    SolverControl solver_control(vel_max_its, 1e-12*rhs_E.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    if(HYPERBOLIC_stage == 1) {
      E_tmp_2.equ(1.0, E_old);
      cg.solve(euler_matrix, E_tmp_2, rhs_E, PreconditionIdentity());
    }
    else {
      E_curr.equ(1.0, E_tmp_2);
      cg.solve(euler_matrix, E_curr, rhs_E, PreconditionIdentity());
    }
  }


  // @sect4{<code>NavierStokesProjection::project_velocity</code>}

  // This implements the projection of the velocity in the hyperbolic part
  //
  template<int dim>
  void NavierStokesProjection<dim>::project_velocity() {
    TimerOutput::Scope t(time_table, "Project velocity");

    const std::vector<unsigned int> tmp = {0};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);

    euler_matrix.set_NS_stage(4);

    euler_matrix.vmult_rhs_velocity_projection(rhs_u, {rho_curr, u_curr});

    SolverControl solver_control(vel_max_its, 1e-12*rhs_u.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    u_proj.equ(1.0, u_old);
    cg.solve(euler_matrix, u_proj, rhs_u, PreconditionIdentity());
  }


  template<int dim>
  void NavierStokesProjection<dim>::project_pressure() {
    TimerOutput::Scope t(time_table, "Project pressure");

    const std::vector<unsigned int> tmp = {1};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);

    euler_matrix.set_NS_stage(5);

    if(HYPERBOLIC_stage == 1)
      euler_matrix.vmult_rhs_pressure_projection(rhs_pres, {rho_tmp_2, u_tmp_2, E_tmp_2});
    else
      euler_matrix.vmult_rhs_pressure_projection(rhs_pres, {rho_curr, u_curr, E_curr});

    SolverControl solver_control(vel_max_its, 1e-12*rhs_pres.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    if(HYPERBOLIC_stage == 1) {
      pres_tmp_2.equ(1.0, pres_old);
      cg.solve(euler_matrix, pres_tmp_2, rhs_pres, PreconditionIdentity());
    }
    else {
      pres_old.equ(1.0, pres_tmp_2);
      cg.solve(euler_matrix, pres_old, rhs_pres, PreconditionIdentity());
    }
  }


  // @sect4{ <code>NavierStokesProjection::output_results</code> }

  // This method plots the current solution. The main difficulty is that we want
  // to create a single output file that contains the data for all velocity
  // components and the pressure. On the other hand, velocities and the pressure
  // live on separate DoFHandler objects, and
  // so can't be written to the same file using a single DataOut object. As a
  // consequence, we have to work a bit harder to get the various pieces of data
  // into a single DoFHandler object, and then use that to drive graphical
  // output.
  //
  template<int dim>
  void NavierStokesProjection<dim>::output_results(const unsigned int step) {
    TimerOutput::Scope t(time_table, "Output results");

    DataOut<dim> data_out;

    rho_old.update_ghost_values();
    data_out.add_data_vector(dof_handler_density, rho_old, "rho", {DataComponentInterpretation::component_is_scalar});
    std::vector<std::string> velocity_names(dim, "u");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
    u_old.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_old, velocity_names, component_interpretation_velocity);
    E_old.update_ghost_values();
    data_out.add_data_vector(dof_handler_temperature, E_old, "E", {DataComponentInterpretation::component_is_scalar});

    data_out.build_patches();
    const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
  }


  // The following function is used in determining the maximal velocity
  // in order to compute the CFL
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_velocity() {
    project_velocity();

    VectorTools::integrate_difference(dof_handler_velocity, u_proj, ZeroFunction<dim>(dim),
                                      Linfty_error_per_cell_vel, quadrature_velocity, VectorTools::Linfty_norm);
    const double res = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_vel, VectorTools::Linfty_norm);

    return res;
  }


  // @sect4{ <code>NavierStokesProjection::run</code> }

  // This is the time marching function, which starting at <code>t_0</code>
  // advances in time using the projection method with time step <code>dt</code>
  // until <code>T</code>.
  //
  // Its second parameter, <code>verbose</code> indicates whether the function
  // should output information what it is doing at any given moment:
  // we use the ConditionalOStream class to do that for us.
  //
  template<int dim>
  void NavierStokesProjection<dim>::run(const bool verbose, const unsigned int output_interval) {
    ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    output_results(0);
    double time = t_0;
    unsigned int n = 0;
    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;

      //--- First stage of HYPERBOLIC operator
      euler_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);
      verbose_cout << "  Update density stage 1" << std::endl;
      update_density();
      verbose_cout << "  Update velocity stage 1" << std::endl;
      update_momentum();
      verbose_cout << "  Update total energy stage 1" << std::endl;
      update_energy();
      HYPERBOLIC_stage = 2; //--- Flag to pass at second stage

      //--- Second stage of HYPERBOLIC operator
      euler_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);
      verbose_cout << "  Update density stage 2" << std::endl;
      update_density();
      verbose_cout << "  Update velocity stage 2" << std::endl;
      update_momentum();
      verbose_cout << "  Update total energy stage 2" << std::endl;
      update_energy();
      HYPERBOLIC_stage = 1; //--- Flag to pass at first stage at next step

      //--- Update for next step
      rho_old.equ(1.0, rho_curr);
      u_old.equ(1.0, u_curr);
      E_old.equ(1.0, E_curr);
      const double max_velocity = get_maximal_velocity();
      pcout<< "Maximal velocity = " << max_velocity << std::endl;
      pcout << "CFL_u = " << dt*max_velocity*EquationData::degree_u*
                             std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;

      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution final" << std::endl;
        output_results(n);
      }
      //if(time > 0.1*T && get_maximal_difference() < 1e-7)
      //  break;
      if(T - time < dt && T - time > 1e-10) {
        dt = T - time;
        euler_matrix.set_dt(dt);
      }
    }
    if(n % output_interval != 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }
  }

} // namespace Step35


// @sect3{ The main function }

// The main function looks very much like in all the other tutorial programs, so
// there is little to comment on here:
int main(int argc, char *argv[]) {
  try {
    using namespace Step35;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    NavierStokesProjection<2> test(data);
    test.run(data.verbose, data.output_interval);

    if(curr_rank == 0)
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;

    return 0;
  }
  catch(std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

}
