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
      const double t = this->get_time();
      const double beta = 5.0;

      Point<dim> x0;
      x0[0] = 5.;
      const double radius_sqr  = (p - x0).norm_square() - 2.0*(p[0] - x0[0])*t + t*t;
      const double factor      = beta/(2.0*numbers::PI)*std::exp(1.0 - radius_sqr);
      if(component == 0)
        return 1.0 - factor*(p[1] - x0[1]);
      else
        return factor*(p[0] - t - x0[0]);
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
    class Pressure: public Function<dim> {
    public:
      Pressure(const double initial_time = 0.0);

      virtual double value(const Point<dim>&  p,
                           const unsigned int component = 0) const override;
    };


    template<int dim>
    Pressure<dim>::Pressure(const double initial_time): Function<dim>(1, initial_time) {}


    template<int dim>
    double Pressure<dim>::value(const Point<dim>&  p, const unsigned int component) const {
      (void)component;
      AssertIndexRange(component, 1);

      const double t = this->get_time();
      const double beta = 5.0;

      Point<dim> x0;
      x0[0] = 5.;
      const double radius_sqr  = (p - x0).norm_square() - 2.0*(p[0] - x0[0])*t + t*t;
      const double factor      = beta/(2.0*numbers::PI)*std::exp(1.0 - radius_sqr);
      const double density_log = std::log2(std::abs(1.0 - (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv*0.25*factor*factor));
      return std::exp2(density_log*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)));
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

      const double t = this->get_time();
      const double beta = 5.0;

      Point<dim> x0;
      x0[0] = 5.;
      const double radius_sqr  = (p - x0).norm_square() - 2.0*(p[0] - x0[0])*t + t*t;
      const double factor      = beta/(2.0*numbers::PI)*std::exp(1.0 - radius_sqr);
      const double density_log = std::log2(std::abs(1.0 - (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv*0.25*factor*factor));
      return std::exp2(density_log*(1.0/(EquationData::Cp_Cv - 1.0)));
    }


    // We do the same for the energy (since it is a scalar field) we can derive
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

      const double t = this->get_time();
      const double beta = 5.0;

      Point<dim> x0;
      x0[0] = 5.;
      const double radius_sqr  = (p - x0).norm_square() - 2.0*(p[0] - x0[0])*t + t*t;
      const double factor      = beta/(2.0*numbers::PI)*std::exp(1.0 - radius_sqr);
      const double density_log = std::log2(std::abs(1.0 - (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv*0.25*factor*factor));
      const double density     = std::exp2(density_log*(1.0/(EquationData::Cp_Cv - 1.0)));
      const double u           = 1.0 - factor*(p[1] - x0[1]);
      const double v           = factor*(p[0] - t - x0[0]);
      const double pressure    = std::exp2(density_log*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)));
      return pressure/(density*(EquationData::Cp_Cv - 1.0)) + 0.5*(u*u + v*v);
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

    void set_u_fixed(const Vec& src);

    void advance_rho_boundary_time(const double time_advance);

    void advance_pres_boundary_time(const double time_advance);

    void advance_u_boundary_time(const double time_advance);

    void vmult_rhs_rho_projection(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_velocity_update(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_pressure(Vec& dst, const Vec& src) const;

    void vmult_enthalpy(Vec& dst, const Vec& src) const;

    Number compute_max_celerity(const std::vector<Vec>& src) const;

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
    mutable unsigned int NS_stage;

    virtual void apply_add(Vec& dst, const Vec& src) const override;

  private:
    const double theta_p = 1.0;
    const double C_p = 1.0*(fe_degree_rho + 1)*(fe_degree_rho + 1);

    Vec rho_for_fixed,
        pres_fixed,
        u_fixed;

    EquationData::Density<dim>  rho_boundary;
    EquationData::Pressure<dim> pres_boundary;
    EquationData::Velocity<dim> u_boundary;

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

    void assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_enthalpy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_enthalpy(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_rhs_cell_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                                    Vec&                                         dst,
                                                    const std::vector<Vec>&                      src,
                                                    const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const Vec&                                   src,
                                                const std::pair<unsigned int, unsigned int>& face_range) const;
  };


  // Default constructor
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u, n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  HYPERBOLICOperator(): MatrixFreeOperators::Base<dim, Vec>(), Ma(), Fr(), dt(), gamma(2.0 - std::sqrt(2.0)),
                        a21(gamma), a22(0.0), a31((2.0*gamma - 1.0)/6.0), a32((7.0 - 2.0*gamma)/6.0), a33(0.0),
                        a21_tilde(0.5*gamma), a22_tilde(0.5*gamma), a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                        a33_tilde(1.0 - std::sqrt(2)/2.0), HYPERBOLIC_stage(1), NS_stage(1), rho_boundary(), pres_boundary(),
                        u_boundary() {}


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
                                                             a33_tilde(1.0 - std::sqrt(2)/2.0), HYPERBOLIC_stage(1), NS_stage(1),
                                                             rho_boundary(data.initial_time), pres_boundary(data.initial_time),
                                                             u_boundary(data.initial_time) {}


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
    AssertIndexRange(stage, 6);
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


  // Setter of velocity for fixed point
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_u_fixed(const Vec& src) {
    u_fixed = src;
    u_fixed.update_ghost_values();
  }


  // Advance density boundary time
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  advance_rho_boundary_time(const double time_advance) {
    rho_boundary.advance_time(time_advance);
  }


  // Advance pressure boundary time
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  advance_pres_boundary_time(const double time_advance) {
    pres_boundary.advance_time(time_advance);
  }


  // Advance velocity boundary time
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  advance_u_boundary_time(const double time_advance) {
    u_boundary.advance_time(time_advance);
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
          phi.submit_gradient(a21*dt*phi_rho_old.get_value(q)*phi_u_old.get_value(q), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi(data, 2),
                                                                     phi_rho_old(data, 2),
                                                                     phi_rho_tmp_2(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0);

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
          phi.submit_gradient(a31*dt*phi_rho_old.get_value(q)*phi_u_old.get_value(q) +
                              a32*dt*phi_rho_tmp_2.get_value(q)*phi_u_tmp_2.get_value(q), q);
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
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_p(data, true, 2),
                                                                         phi_m(data, false, 2),
                                                                         phi_rho_old_p(data, true, 2),
                                                                         phi_rho_old_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old_p(data, true, 1),
                                                                     phi_pres_old_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus   = phi_p.get_normal_vector(q);

          const auto& avg_flux = 0.5*(phi_rho_old_p.get_value(q)*phi_u_old_p.get_value(q) +
                                      phi_rho_old_m.get_value(q)*phi_u_old_m.get_value(q));
          const auto  lambda   = std::max(std::sqrt(scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q))) +
                                          std::sqrt(EquationData::Cp_Cv*phi_pres_old_p.get_value(q)/phi_rho_old_p.get_value(q)),
                                          std::sqrt(scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q))) +
                                          std::sqrt(EquationData::Cp_Cv*phi_pres_old_m.get_value(q)/phi_rho_old_m.get_value(q)));
          const auto& jump_rho_old = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          phi_p.submit_value(-a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho_old), q);
          phi_m.submit_value(a21*dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho_old), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_p(data, true, 2),
                                                                         phi_m(data, false, 2),
                                                                         phi_rho_old_p(data, true, 2),
                                                                         phi_rho_old_m(data, false, 2),
                                                                         phi_rho_tmp_2_p(data, true, 2),
                                                                         phi_rho_tmp_2_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old_p(data, true, 1),
                                                                     phi_pres_old_m(data, false, 1),
                                                                     phi_pres_tmp_2_p(data, true, 1),
                                                                     phi_pres_tmp_2_m(data, false, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& avg_flux_old   = 0.5*(phi_rho_old_p.get_value(q)*phi_u_old_p.get_value(q) +
                                            phi_rho_old_m.get_value(q)*phi_u_old_m.get_value(q));
          const auto  lambda_old     = std::max(std::sqrt(scalar_product(phi_u_old_p.get_value(q), phi_u_old_p.get_value(q))) +
                                                std::sqrt(EquationData::Cp_Cv*phi_pres_old_p.get_value(q)/phi_rho_old_p.get_value(q)),
                                                std::sqrt(scalar_product(phi_u_old_m.get_value(q), phi_u_old_m.get_value(q))) +
                                                std::sqrt(EquationData::Cp_Cv*phi_pres_old_m.get_value(q)/phi_rho_old_m.get_value(q)));
          const auto& jump_rho_old   = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          const auto& avg_flux_tmp_2 = 0.5*(phi_rho_tmp_2_p.get_value(q)*phi_u_tmp_2_p.get_value(q) +
                                            phi_rho_tmp_2_m.get_value(q)*phi_u_tmp_2_m.get_value(q));
          const auto  lambda_tmp_2   = std::max(std::sqrt(scalar_product(phi_u_tmp_2_p.get_value(q), phi_u_tmp_2_p.get_value(q))) +
                                                std::sqrt(EquationData::Cp_Cv*phi_pres_tmp_2_p.get_value(q)/phi_rho_tmp_2_p.get_value(q)),
                                                std::sqrt(scalar_product(phi_u_tmp_2_m.get_value(q), phi_u_tmp_2_m.get_value(q))) +
                                                std::sqrt(EquationData::Cp_Cv*phi_pres_tmp_2_m.get_value(q)/phi_rho_tmp_2_m.get_value(q)));
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
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi(data, true, 2),
                                                                         phi_rho_old(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old(data, true, 1);
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_old_D        = VectorizedArray<Number>();
          auto        pres_old_D       = VectorizedArray<Number>();
          auto        u_old_D          = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_old_D[v] = pres_boundary.value(point);
            rho_old_D[v]  = rho_boundary.value(point);
            for(unsigned int d = 0; d < dim; ++d)
              u_old_D[d][v] = u_boundary.value(point, d);
          }

          const auto& avg_flux = 0.5*(phi_rho_old.get_value(q)*phi_u_old.get_value(q) + rho_old_D*u_old_D);
          const auto  lambda   = std::max(std::sqrt(scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q))) +
                                          std::sqrt(EquationData::Cp_Cv*phi_pres_old.get_value(q)/phi_rho_old.get_value(q)),
                                          std::sqrt(scalar_product(u_old_D, u_old_D)) +
                                          std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
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
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old(data, true, 1),
                                                                     phi_pres_tmp_2(data, true, 1);

      auto pres_boundary_tmp_2 = pres_boundary;
      auto rho_boundary_tmp_2  = rho_boundary;
      auto u_boundary_tmp_2    = u_boundary;
      pres_boundary_tmp_2.advance_time(gamma*dt);
      rho_boundary_tmp_2.advance_time(gamma*dt);
      u_boundary_tmp_2.advance_time(gamma*dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_old_D        = VectorizedArray<Number>();
          auto        pres_old_D       = VectorizedArray<Number>();
          auto        u_old_D          = Tensor<1, dim, VectorizedArray<Number>>();
          auto        rho_tmp_2_D      = VectorizedArray<Number>();
          auto        pres_tmp_2_D     = VectorizedArray<Number>();
          auto        u_tmp_2_D        = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_old_D[v]   = pres_boundary.value(point);
            pres_tmp_2_D[v] = pres_boundary_tmp_2.value(point);
            rho_old_D[v]    = rho_boundary.value(point);
            rho_tmp_2_D[v]  = rho_boundary_tmp_2.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v]   = u_boundary.value(point, d);
              u_tmp_2_D[d][v] = u_boundary_tmp_2.value(point, d);
            }
          }

          const auto& avg_flux_old   = 0.5*(phi_rho_old.get_value(q)*phi_u_old.get_value(q) + rho_old_D*u_old_D);
          const auto  lambda_old     = std::max(std::sqrt(scalar_product(phi_u_old.get_value(q), phi_u_old.get_value(q))) +
                                                std::sqrt(EquationData::Cp_Cv*phi_pres_old.get_value(q)/phi_rho_old.get_value(q)),
                                                std::sqrt(scalar_product(u_old_D, u_old_D)) +
                                                std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_rho_old   = phi_rho_old.get_value(q) - rho_old_D;

          const auto& avg_flux_tmp_2 = 0.5*(phi_rho_tmp_2.get_value(q)*phi_u_tmp_2.get_value(q) + rho_tmp_2_D*u_tmp_2_D);
          const auto  lambda_tmp_2   = std::max(std::sqrt(scalar_product(phi_u_tmp_2.get_value(q), phi_u_tmp_2.get_value(q))) +
                                                std::sqrt(EquationData::Cp_Cv*phi_pres_tmp_2.get_value(q)/phi_rho_tmp_2.get_value(q)),
                                                std::sqrt(scalar_product(u_tmp_2_D, u_tmp_2_D)) +
                                                std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));
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
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi(data, 2, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1),
                                                                 phi_pres_old(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_fixed(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, 2),
                                                                     phi_rho_tmp_2(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[4], true, false);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old   = phi_rho_old.get_value(q);
          const auto& u_old     = phi_u_old.get_value(q);
          const auto& pres_old  = phi_pres_old.get_value(q);
          const auto& rho_tmp_2 = phi_rho_tmp_2.get_value(q);
          const auto& u_fixed   = phi_u_fixed.get_value(q);
          const auto& E_old     = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_old/rho_old) + 0.5*scalar_product(u_old, u_old);

          phi.submit_value(rho_old*E_old -
                           0.5*rho_tmp_2*scalar_product(u_fixed, u_fixed) -
                           0.0*a21*dt*Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1], q);
          phi.submit_gradient(0.5*a21*dt*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old +
                              a21_tilde*dt*((rho_old*(E_old - 0.5*scalar_product(u_old, u_old)) + pres_old)*u_old), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1),
                                                                 phi_pres_old(data, 1),
                                                                 phi_pres_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0),
                                                                   phi_u_fixed(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, 2),
                                                                     phi_rho_tmp_2(data, 2),
                                                                     phi_rho_curr(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);
        phi_rho_curr.reinit(cell);
        phi_rho_curr.gather_evaluate(src[6], true, false);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[7], true, false);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old    = phi_rho_old.get_value(q);
          const auto& u_old      = phi_u_old.get_value(q);
          const auto& pres_old   = phi_pres_old.get_value(q);
          const auto& rho_tmp_2  = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2    = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2 = phi_pres_old.get_value(q);
          const auto& rho_curr   = phi_rho_curr.get_value(q);
          const auto& u_fixed    = phi_u_fixed.get_value(q);
          const auto& E_old      = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_old/rho_old) + 0.5*scalar_product(u_old, u_old);
          const auto& E_tmp_2    = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_tmp_2/rho_tmp_2) + 0.5*scalar_product(u_tmp_2, u_tmp_2);

          phi.submit_value(rho_old*E_old -
                           0.5*rho_curr*scalar_product(u_fixed, u_fixed) -
                           0.0*a31*dt*Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1] -
                           0.0*a32*dt*Ma*Ma/(Fr*Fr)*rho_tmp_2*u_tmp_2[dim - 1], q);
          phi.submit_gradient(0.5*a31*dt*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old +
                              a31_tilde*dt*((rho_old*(E_old - 0.5*scalar_product(u_old, u_old)) + pres_old)*u_old) +
                              0.5*a32*dt*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)*rho_tmp_2*u_tmp_2 +
                              a32_tilde*dt*((rho_tmp_2*(E_tmp_2 - 0.5*scalar_product(u_tmp_2, u_tmp_2)) + pres_tmp_2)*u_tmp_2), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_p(data, true, 1),
                                                                     phi_m(data, false, 1),
                                                                     phi_pres_old_p(data, true, 1),
                                                                     phi_pres_old_m(data, false, 1),
                                                                     phi_pres_fixed_p(data, true, 1),
                                                                     phi_pres_fixed_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_fixed_p(data, true, 0),
                                                                       phi_u_fixed_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old_p(data, true, 2),
                                                                         phi_rho_old_m(data, false, 2),
                                                                         phi_rho_tmp_2_p(data, true, 2),
                                                                         phi_rho_tmp_2_m(data, false, 2);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_fixed_p.reinit(face);
        phi_u_fixed_p.gather_evaluate(src[4], true, false);
        phi_u_fixed_m.reinit(face);
        phi_u_fixed_m.gather_evaluate(src[4], true, false);
        phi_pres_fixed_p.reinit(face);
        phi_pres_fixed_p.gather_evaluate(src[5], true, false);
        phi_pres_fixed_m.reinit(face);
        phi_pres_fixed_m.gather_evaluate(src[5], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus           = phi_p.get_normal_vector(q);

          const auto& rho_old_p        = phi_rho_old_p.get_value(q);
          const auto& rho_old_m        = phi_rho_old_m.get_value(q);
          const auto& u_old_p          = phi_u_old_p.get_value(q);
          const auto& u_old_m          = phi_u_old_m.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old_p,u_old_p)*rho_old_p*u_old_p +
                                              0.5*scalar_product(u_old_m,u_old_m)*rho_old_m*u_old_m);
          const auto& pres_old_p       = phi_pres_old_p.get_value(q);
          const auto& pres_old_m       = phi_pres_old_m.get_value(q);
          const auto& E_old_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p/rho_old_p + 0.5*scalar_product(u_old_p, u_old_p);
          const auto& E_old_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m/rho_old_m + 0.5*scalar_product(u_old_m, u_old_m);
          const auto& avg_enthalpy_old = 0.5*(((E_old_p - 0.5*scalar_product(u_old_p,u_old_p))*rho_old_p + pres_old_p)*u_old_p +
                                              ((E_old_m - 0.5*scalar_product(u_old_m,u_old_m))*rho_old_m + pres_old_m)*u_old_m);
          const auto& lambda_old       = std::max(scalar_product(u_old_p, u_old_p) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_old_p/rho_old_p),
                                                  scalar_product(u_old_m, u_old_m) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_old_m/rho_old_m));
          const auto& jump_rhoE_old    = rho_old_p*E_old_p - rho_old_m*E_old_m;

          const auto& rho_tmp_2_p      = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m      = phi_rho_tmp_2_m.get_value(q);
          const auto& u_fixed_p        = phi_u_fixed_p.get_value(q);
          const auto& u_fixed_m        = phi_u_fixed_m.get_value(q);
          const auto& pres_fixed_p     = phi_pres_fixed_p.get_value(q);
          const auto& pres_fixed_m     = phi_pres_fixed_m.get_value(q);
          const auto& lambda_fixed     = std::max(scalar_product(u_fixed_p, u_fixed_p) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_fixed_p/rho_tmp_2_p),
                                                  scalar_product(u_fixed_m, u_fixed_m) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_fixed_m/rho_tmp_2_m));
          const auto& jump_rhok_fixed  = rho_tmp_2_p*0.5*scalar_product(u_fixed_p, u_fixed_p) -
                                         rho_tmp_2_m*0.5*scalar_product(u_fixed_m, u_fixed_m);

          phi_p.submit_value(-a21*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus)
                             -a21_tilde*dt*scalar_product(avg_enthalpy_old, n_plus)
                             -a21*dt*0.5*lambda_old*jump_rhoE_old
                             -a22_tilde*dt*0.5*lambda_fixed*jump_rhok_fixed, q);
          phi_m.submit_value(a21*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                             a21_tilde*dt*scalar_product(avg_enthalpy_old, n_plus) +
                             a21*dt*0.5*lambda_old*jump_rhoE_old +
                             a22_tilde*dt*0.5*lambda_fixed*jump_rhok_fixed, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_p(data, true, 1),
                                                                     phi_m(data, false, 1),
                                                                     phi_pres_old_p(data, true, 1),
                                                                     phi_pres_old_m(data, false, 1),
                                                                     phi_pres_tmp_2_p(data, true, 1),
                                                                     phi_pres_tmp_2_m(data, false, 1),
                                                                     phi_pres_fixed_p(data, true, 1),
                                                                     phi_pres_fixed_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0),
                                                                       phi_u_fixed_p(data, true, 0),
                                                                       phi_u_fixed_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old_p(data, true, 2),
                                                                         phi_rho_old_m(data, false, 2),
                                                                         phi_rho_tmp_2_p(data, true, 2),
                                                                         phi_rho_tmp_2_m(data, false, 2),
                                                                         phi_rho_curr_p(data, true, 2),
                                                                         phi_rho_curr_m(data, false, 2);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], true, false);
        phi_rho_curr_p.reinit(face);
        phi_rho_curr_p.gather_evaluate(src[6], true, false);
        phi_rho_curr_m.reinit(face);
        phi_rho_curr_m.gather_evaluate(src[6], true, false);
        phi_u_fixed_p.reinit(face);
        phi_u_fixed_p.gather_evaluate(src[7], true, false);
        phi_u_fixed_m.reinit(face);
        phi_u_fixed_m.gather_evaluate(src[7], true, false);
        phi_pres_fixed_p.reinit(face);
        phi_pres_fixed_p.gather_evaluate(src[8], true, false);
        phi_pres_fixed_m.reinit(face);
        phi_pres_fixed_m.gather_evaluate(src[8], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus           = phi_p.get_normal_vector(q);

          const auto& rho_old_p        = phi_rho_old_p.get_value(q);
          const auto& rho_old_m        = phi_rho_old_m.get_value(q);
          const auto& u_old_p          = phi_u_old_p.get_value(q);
          const auto& u_old_m          = phi_u_old_m.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old_p,u_old_p)*rho_old_p*u_old_p +
                                              0.5*scalar_product(u_old_m,u_old_m)*rho_old_m*u_old_m);
          const auto& pres_old_p       = phi_pres_old_p.get_value(q);
          const auto& pres_old_m       = phi_pres_old_m.get_value(q);
          const auto& E_old_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p/rho_old_p + 0.5*scalar_product(u_old_p, u_old_p);
          const auto& E_old_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m/rho_old_m + 0.5*scalar_product(u_old_m, u_old_m);
          const auto& avg_enthalpy_old = 0.5*(((E_old_p - 0.5*scalar_product(u_old_p,u_old_p))*rho_old_p + pres_old_p)*u_old_p +
                                              ((E_old_m - 0.5*scalar_product(u_old_m,u_old_m))*rho_old_m + pres_old_m)*u_old_m);
          const auto& lambda_old       = std::max(scalar_product(u_old_p, u_old_p) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_old_p/rho_old_p),
                                                  scalar_product(u_old_m, u_old_m) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_old_m/rho_old_m));
          const auto& jump_rhoE_old    = rho_old_p*E_old_p - rho_old_m*E_old_m;

          const auto& rho_tmp_2_p        = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m        = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p          = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m          = phi_u_tmp_2_m.get_value(q);
          const auto& avg_kinetic_tmp_2  = 0.5*(0.5*scalar_product(u_tmp_2_p,u_tmp_2_p)*rho_tmp_2_p*u_tmp_2_p +
                                                0.5*scalar_product(u_tmp_2_m,u_tmp_2_m)*rho_tmp_2_m*u_tmp_2_m);
          const auto& pres_tmp_2_p       = phi_pres_tmp_2_p.get_value(q);
          const auto& pres_tmp_2_m       = phi_pres_tmp_2_m.get_value(q);
          const auto& E_tmp_2_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_p/rho_tmp_2_p
                                         + 0.5*scalar_product(u_tmp_2_p, u_tmp_2_p);
          const auto& E_tmp_2_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_m/rho_tmp_2_m
                                         + 0.5*scalar_product(u_tmp_2_m, u_tmp_2_m);
          const auto& avg_enthalpy_tmp_2 = 0.5*
                                           (((E_tmp_2_p - 0.5*scalar_product(u_tmp_2_p,u_tmp_2_p))*rho_tmp_2_p + pres_tmp_2_p)*u_tmp_2_p +
                                            ((E_tmp_2_m - 0.5*scalar_product(u_tmp_2_m,u_tmp_2_m))*rho_tmp_2_m + pres_tmp_2_m)*u_tmp_2_m);
          const auto& lambda_tmp_2       = std::max(scalar_product(u_tmp_2_p, u_tmp_2_p) +
                                                    std::sqrt(EquationData::Cp_Cv*pres_tmp_2_p/rho_tmp_2_p),
                                                    scalar_product(u_tmp_2_m, u_tmp_2_m) +
                                                    std::sqrt(EquationData::Cp_Cv*pres_tmp_2_m/rho_tmp_2_m));
          const auto& jump_rhoE_tmp_2    = rho_tmp_2_p*E_tmp_2_p - rho_tmp_2_m*E_tmp_2_m;

          const auto& rho_curr_p         = phi_rho_curr_p.get_value(q);
          const auto& rho_curr_m         = phi_rho_curr_m.get_value(q);
          const auto& u_fixed_p          = phi_u_fixed_p.get_value(q);
          const auto& u_fixed_m          = phi_u_fixed_m.get_value(q);
          const auto& pres_fixed_p       = phi_pres_fixed_p.get_value(q);
          const auto& pres_fixed_m       = phi_pres_fixed_m.get_value(q);
          const auto& lambda_fixed       = std::max(scalar_product(u_fixed_p, u_fixed_p) +
                                                    std::sqrt(EquationData::Cp_Cv*pres_fixed_p/rho_curr_p),
                                                    scalar_product(u_fixed_m, u_fixed_m) +
                                                    std::sqrt(EquationData::Cp_Cv*pres_fixed_m/rho_curr_m));
          const auto& jump_rhok_fixed    = rho_curr_p*0.5*scalar_product(u_fixed_p, u_fixed_p) -
                                           rho_curr_m*0.5*scalar_product(u_fixed_m, u_fixed_m);

          phi_p.submit_value(-a31*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus)
                             -a31_tilde*dt*scalar_product(avg_enthalpy_old, n_plus)
                             -a31*dt*0.5*lambda_old*jump_rhoE_old
                             -a32*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus)
                             -a32_tilde*dt*scalar_product(avg_enthalpy_tmp_2, n_plus)
                             -a32*dt*0.5*lambda_tmp_2*jump_rhoE_tmp_2
                             -a33_tilde*dt*0.5*lambda_fixed*jump_rhok_fixed, q);
          phi_m.submit_value(a31*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                             a31_tilde*dt*scalar_product(avg_enthalpy_old, n_plus) +
                             a31*dt*0.5*lambda_old*jump_rhoE_old +
                             a32*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus) +
                             a32_tilde*dt*scalar_product(avg_enthalpy_tmp_2, n_plus) +
                             a32*dt*0.5*lambda_tmp_2*jump_rhoE_tmp_2 +
                             a33_tilde*dt*0.5*lambda_fixed*jump_rhok_fixed, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, true, 1),
                                                                     phi_pres_old(data, true, 1),
                                                                     phi_pres_fixed(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0),
                                                                       phi_u_fixed(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, true, 2),
                                                                         phi_rho_tmp_2(data, true, 2);

      auto pres_boundary_tmp_2 = pres_boundary;
      auto rho_boundary_tmp_2  = rho_boundary;
      auto u_boundary_tmp_2    = u_boundary;
      pres_boundary_tmp_2.advance_time(gamma*dt);
      rho_boundary_tmp_2.advance_time(gamma*dt);
      u_boundary_tmp_2.advance_time(gamma*dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_fixed.reinit(face);
        phi_u_fixed.gather_evaluate(src[4], true, false);
        phi_pres_fixed.reinit(face);
        phi_pres_fixed.gather_evaluate(src[5], true, false);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_old_D        = VectorizedArray<Number>();
          auto        pres_old_D       = VectorizedArray<Number>();
          auto        u_old_D          = Tensor<1, dim, VectorizedArray<Number>>();
          auto        rho_tmp_2_D      = VectorizedArray<Number>();
          auto        pres_tmp_2_D     = VectorizedArray<Number>();
          auto        u_tmp_2_D        = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_old_D[v]   = pres_boundary.value(point);
            pres_tmp_2_D[v] = pres_boundary_tmp_2.value(point);
            rho_old_D[v]    = rho_boundary.value(point);
            rho_tmp_2_D[v]  = rho_boundary_tmp_2.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v]   = u_boundary.value(point, d);
              u_tmp_2_D[d][v] = u_boundary_tmp_2.value(point, d);
            }
          }

          const auto& rho_old          = phi_rho_old.get_value(q);
          const auto& u_old            = phi_u_old.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old,u_old)*rho_old*u_old +
                                              0.5*scalar_product(u_old_D,u_old_D)*rho_old_D*u_old_D);
          const auto& pres_old         = phi_pres_old.get_value(q);
          const auto& E_old            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old/rho_old + 0.5*scalar_product(u_old, u_old);
          const auto& E_old_D          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_D/rho_old_D + 0.5*scalar_product(u_old_D, u_old_D);
          const auto& avg_enthalpy_old = 0.5*(((E_old - 0.5*scalar_product(u_old,u_old))*rho_old + pres_old)*u_old +
                                              ((E_old_D - 0.5*scalar_product(u_old_D,u_old_D))*rho_old_D + pres_old_D)*u_old_D);
          const auto& lambda_old       = std::max(scalar_product(u_old, u_old) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_old/rho_old),
                                                  scalar_product(u_old_D, u_old_D) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_rhoE_old    = rho_old*E_old - rho_old_D*E_old_D;

          const auto& rho_tmp_2        = phi_rho_tmp_2.get_value(q);
          const auto& u_fixed          = phi_u_fixed.get_value(q);
          const auto& pres_fixed       = phi_pres_fixed.get_value(q);
          const auto& lambda_fixed     = std::max(scalar_product(u_fixed, u_fixed) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_fixed/rho_tmp_2),
                                                  scalar_product(u_tmp_2_D, u_tmp_2_D) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));
          const auto& E_tmp_2_D        = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_D/rho_tmp_2_D
                                       + 0.5*rho_tmp_2_D*scalar_product(u_tmp_2_D, u_tmp_2_D);
          const auto& jump_rhok_fixed  = rho_tmp_2*0.5*scalar_product(u_fixed, u_fixed) -
                                         rho_tmp_2_D*E_tmp_2_D;
          const auto& enthalpy_fixed   = ((E_tmp_2_D - 0.5*scalar_product(u_tmp_2_D,u_tmp_2_D))*rho_tmp_2_D + pres_tmp_2_D)*u_tmp_2_D;

          phi.submit_value(-a21*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus)
                           -a21_tilde*dt*scalar_product(avg_enthalpy_old, n_plus)
                           -a21*dt*0.5*lambda_old*jump_rhoE_old
                           -a22_tilde*dt*0.5*lambda_fixed*jump_rhok_fixed
                           -a22_tilde*dt*0.5*scalar_product(enthalpy_fixed, n_plus), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, true, 1),
                                                                     phi_pres_old(data, true, 1),
                                                                     phi_pres_tmp_2(data, true, 1),
                                                                     phi_pres_fixed(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0),
                                                                       phi_u_fixed(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, true, 2),
                                                                         phi_rho_tmp_2(data, true, 2),
                                                                         phi_rho_curr(data, true, 2);

      auto pres_boundary_tmp_2 = pres_boundary;
      auto rho_boundary_tmp_2  = rho_boundary;
      auto u_boundary_tmp_2    = u_boundary;
      pres_boundary_tmp_2.advance_time(gamma*dt);
      rho_boundary_tmp_2.advance_time(gamma*dt);
      u_boundary_tmp_2.advance_time(gamma*dt);
      auto pres_boundary_curr = pres_boundary;
      auto rho_boundary_curr  = rho_boundary;
      auto u_boundary_curr    = u_boundary;
      pres_boundary_curr.advance_time(dt);
      rho_boundary_curr.advance_time(dt);
      u_boundary_curr.advance_time(dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);
        phi_rho_curr.reinit(face);
        phi_rho_curr.gather_evaluate(src[6], true, false);
        phi_u_fixed.reinit(face);
        phi_u_fixed.gather_evaluate(src[7], true, false);
        phi_pres_fixed.reinit(face);
        phi_pres_fixed.gather_evaluate(src[8], true, false);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_old_D        = VectorizedArray<Number>();
          auto        pres_old_D       = VectorizedArray<Number>();
          auto        u_old_D          = Tensor<1, dim, VectorizedArray<Number>>();
          auto        rho_tmp_2_D      = VectorizedArray<Number>();
          auto        pres_tmp_2_D     = VectorizedArray<Number>();
          auto        u_tmp_2_D        = Tensor<1, dim, VectorizedArray<Number>>();
          auto        rho_curr_D       = VectorizedArray<Number>();
          auto        pres_curr_D      = VectorizedArray<Number>();
          auto        u_curr_D         = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_old_D[v]   = pres_boundary.value(point);
            pres_tmp_2_D[v] = pres_boundary_tmp_2.value(point);
            pres_curr_D[v]  = pres_boundary_curr.value(point);
            rho_old_D[v]    = rho_boundary.value(point);
            rho_tmp_2_D[v]  = rho_boundary_tmp_2.value(point);
            rho_curr_D[v]   = rho_boundary_curr.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v]   = u_boundary.value(point, d);
              u_tmp_2_D[d][v] = u_boundary_tmp_2.value(point, d);
              u_curr_D[d][v]  = u_boundary_curr.value(point, d);
            }
          }

          const auto& rho_old          = phi_rho_old.get_value(q);
          const auto& u_old            = phi_u_old.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old,u_old)*rho_old*u_old +
                                              0.5*scalar_product(u_old_D,u_old_D)*rho_old_D*u_old_D);
          const auto& pres_old         = phi_pres_old.get_value(q);
          const auto& E_old            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old/rho_old + 0.5*scalar_product(u_old, u_old);
          const auto& E_old_D          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_D/rho_old_D + 0.5*scalar_product(u_old_D, u_old_D);
          const auto& avg_enthalpy_old = 0.5*(((E_old - 0.5*scalar_product(u_old,u_old))*rho_old + pres_old)*u_old +
                                              ((E_old_D - 0.5*scalar_product(u_old_D,u_old_D))*rho_old_D + pres_old_D)*u_old_D);
          const auto& lambda_old       = std::max(scalar_product(u_old, u_old) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_old/rho_old),
                                                  scalar_product(u_old_D, u_old_D) +
                                                  std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));
          const auto& jump_rhoE_old    = rho_old*E_old - rho_old_D*E_old_D;

          const auto& rho_tmp_2          = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2            = phi_u_tmp_2.get_value(q);
          const auto& avg_kinetic_tmp_2  = 0.5*(0.5*scalar_product(u_tmp_2,u_old)*rho_tmp_2*u_tmp_2 +
                                                0.5*scalar_product(u_tmp_2_D,u_tmp_2_D)*rho_tmp_2_D*u_tmp_2_D);
          const auto& pres_tmp_2         = phi_pres_tmp_2.get_value(q);
          const auto& E_tmp_2            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2/rho_tmp_2
                                         + 0.5*scalar_product(u_tmp_2, u_tmp_2);
          const auto& E_tmp_2_D          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_D/rho_tmp_2_D
                                         + 0.5*scalar_product(u_tmp_2_D, u_tmp_2_D);
          const auto& avg_enthalpy_tmp_2 = 0.5*
                                           (((E_tmp_2 - 0.5*scalar_product(u_tmp_2,u_tmp_2))*rho_tmp_2 + pres_tmp_2)*u_tmp_2 +
                                            ((E_tmp_2_D - 0.5*scalar_product(u_tmp_2_D,u_tmp_2_D))*rho_tmp_2_D + pres_tmp_2_D)*u_tmp_2_D);
          const auto& lambda_tmp_2       = std::max(scalar_product(u_tmp_2, u_tmp_2) +
                                                    std::sqrt(EquationData::Cp_Cv*pres_tmp_2/rho_tmp_2),
                                                    scalar_product(u_tmp_2_D, u_tmp_2_D) +
                                                    std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));
          const auto& jump_rhoE_tmp_2    = rho_tmp_2*E_tmp_2 - rho_tmp_2_D*E_tmp_2_D;

          const auto& rho_curr        = phi_rho_curr.get_value(q);
          const auto& u_fixed         = phi_u_fixed.get_value(q);
          const auto& pres_fixed      = phi_pres_fixed.get_value(q);
          const auto& lambda_fixed    = std::max(scalar_product(u_fixed, u_fixed) +
                                                std::sqrt(EquationData::Cp_Cv*pres_fixed/rho_curr),
                                                scalar_product(u_curr_D, u_curr_D) +
                                                std::sqrt(EquationData::Cp_Cv*pres_curr_D/rho_curr_D));
          const auto& E_curr_D        = 1.0/(EquationData::Cp_Cv - 1.0)*pres_curr_D/rho_curr_D
                                      + 0.5*rho_curr_D*scalar_product(u_curr_D, u_curr_D);
          const auto& jump_rhok_fixed = rho_curr*0.5*scalar_product(u_fixed, u_fixed) -
                                       rho_curr_D*E_curr_D;
          const auto& enthalpy_fixed  = ((E_curr_D - 0.5*scalar_product(u_curr_D,u_tmp_2_D))*rho_curr_D + pres_curr_D)*u_curr_D;
          phi.submit_value(-a31*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus)
                           -a31_tilde*dt*scalar_product(avg_enthalpy_old, n_plus)
                           -a31*dt*0.5*lambda_old*jump_rhoE_old
                           -a32*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus)
                           -a32_tilde*dt*scalar_product(avg_enthalpy_tmp_2, n_plus)
                           -a32*dt*0.5*lambda_tmp_2*jump_rhoE_tmp_2
                           -a33_tilde*dt*0.5*lambda_fixed*jump_rhok_fixed
                           -a33_tilde*dt*0.5*scalar_product(enthalpy_fixed, n_plus), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_pressure,
                     &HYPERBOLICOperator::assemble_rhs_face_term_pressure,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble face term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_face_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_p(data, true, 1),
                                                                   phi_m(data, false, 1),
                                                                   phi_pres_fixed_p(data, true, 1),
                                                                   phi_pres_fixed_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_fixed_p(data, true, 0),
                                                                     phi_u_fixed_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_for_fixed_p(data, true, 2),
                                                                       phi_rho_for_fixed_m(data, false, 2);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_pres_fixed_p.reinit(face);
      phi_pres_fixed_p.gather_evaluate(pres_fixed, true, false);
      phi_pres_fixed_m.reinit(face);
      phi_pres_fixed_m.gather_evaluate(pres_fixed, true, false);
      phi_rho_for_fixed_p.reinit(face);
      phi_rho_for_fixed_p.gather_evaluate(rho_for_fixed, true, false);
      phi_rho_for_fixed_m.reinit(face);
      phi_rho_for_fixed_m.gather_evaluate(rho_for_fixed, true, false);
      phi_u_fixed_p.reinit(face);
      phi_u_fixed_p.gather_evaluate(u_fixed, true, false);
      phi_u_fixed_m.reinit(face);
      phi_u_fixed_m.gather_evaluate(u_fixed, true, false);
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, true, false);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, true, false);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& u_fixed_p       = phi_u_fixed_p.get_value(q);
        const auto& u_fixed_m       = phi_u_fixed_m.get_value(q);
        const auto& pres_fixed_p    = phi_pres_fixed_p.get_value(q);
        const auto& pres_fixed_m    = phi_pres_fixed_m.get_value(q);
        const auto& rho_for_fixed_p = phi_rho_for_fixed_p.get_value(q);
        const auto& rho_for_fixed_m = phi_rho_for_fixed_m.get_value(q);
        const auto& lambda_fixed    = std::max(scalar_product(u_fixed_p, u_fixed_p) +
                                               std::sqrt(EquationData::Cp_Cv*pres_fixed_p/rho_for_fixed_p),
                                               scalar_product(u_fixed_m, u_fixed_m) +
                                               std::sqrt(EquationData::Cp_Cv*pres_fixed_m/rho_for_fixed_m));
        const auto& jump_src        = phi_p.get_value(q) - phi_m.get_value(q);

        phi_p.submit_value(coeff*dt*1.0/(EquationData::Cp_Cv - 1.0)*0.5*lambda_fixed*jump_src, q);
        phi_m.submit_value(-coeff*dt*1.0/(EquationData::Cp_Cv - 1.0)*0.5*lambda_fixed*jump_src, q);
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }


  // Assemble boundary term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_boundary_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, true, 1),
                                                                     phi_pres_fixed(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_fixed(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_for_fixed(data, true, 2);

      auto pres_boundary_tmp_2 = pres_boundary;
      auto rho_boundary_tmp_2  = rho_boundary;
      auto u_boundary_tmp_2    = u_boundary;
      pres_boundary_tmp_2.advance_time(gamma*dt);
      rho_boundary_tmp_2.advance_time(gamma*dt);
      u_boundary_tmp_2.advance_time(gamma*dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_pres_fixed.reinit(face);
        phi_pres_fixed.gather_evaluate(pres_fixed, true, false);
        phi_rho_for_fixed.reinit(face);
        phi_rho_for_fixed.gather_evaluate(rho_for_fixed, true, false);
        phi_u_fixed.reinit(face);
        phi_u_fixed.gather_evaluate(u_fixed, true, false);
        phi.reinit(face);
        phi.gather_evaluate(src, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_tmp_2_D      = VectorizedArray<Number>();
          auto        pres_tmp_2_D     = VectorizedArray<Number>();
          auto        u_tmp_2_D        = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_tmp_2_D[v] = pres_boundary_tmp_2.value(point);
            rho_tmp_2_D[v]  = rho_boundary_tmp_2.value(point);
            for(unsigned int d = 0; d < dim; ++d)
              u_tmp_2_D[d][v] = u_boundary_tmp_2.value(point, d);
          }

          const auto& u_fixed       = phi_u_fixed.get_value(q);
          const auto& pres_fixed    = phi_pres_fixed.get_value(q);
          const auto& rho_for_fixed = phi_rho_for_fixed.get_value(q);
          const auto& lambda_fixed  = std::max(scalar_product(u_fixed, u_fixed) +
                                               std::sqrt(EquationData::Cp_Cv*pres_fixed/rho_for_fixed),
                                               scalar_product(u_tmp_2_D, u_tmp_2_D) +
                                               std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));

          phi.submit_value(a22_tilde*dt*1.0/(EquationData::Cp_Cv - 1.0)*0.5*lambda_fixed*phi.get_value(q), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, true, 1),
                                                                     phi_pres_fixed(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_fixed(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_for_fixed(data, true, 2);

      auto pres_boundary_curr = pres_boundary;
      auto rho_boundary_curr  = rho_boundary;
      auto u_boundary_curr    = u_boundary;
      pres_boundary_curr.advance_time(dt);
      rho_boundary_curr.advance_time(dt);
      u_boundary_curr.advance_time(dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_pres_fixed.reinit(face);
        phi_pres_fixed.gather_evaluate(pres_fixed, true, false);
        phi_rho_for_fixed.reinit(face);
        phi_rho_for_fixed.gather_evaluate(rho_for_fixed, true, false);
        phi_u_fixed.reinit(face);
        phi_u_fixed.gather_evaluate(u_fixed, true, false);
        phi.reinit(face);
        phi.gather_evaluate(src, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_curr_D      = VectorizedArray<Number>();
          auto        pres_curr_D     = VectorizedArray<Number>();
          auto        u_curr_D        = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_curr_D[v] = pres_boundary_curr.value(point);
            rho_curr_D[v]  = rho_boundary_curr.value(point);
            for(unsigned int d = 0; d < dim; ++d)
              u_curr_D[d][v] = u_boundary_curr.value(point, d);
          }

          const auto& u_fixed       = phi_u_fixed.get_value(q);
          const auto& pres_fixed    = phi_pres_fixed.get_value(q);
          const auto& rho_for_fixed = phi_rho_for_fixed.get_value(q);
          const auto& lambda_fixed  = std::max(scalar_product(u_fixed, u_fixed) +
                                               std::sqrt(EquationData::Cp_Cv*pres_fixed/rho_for_fixed),
                                               scalar_product(u_curr_D, u_curr_D) +
                                               std::sqrt(EquationData::Cp_Cv*pres_curr_D/rho_curr_D));

          phi.submit_value(a33_tilde*dt*1.0/(EquationData::Cp_Cv - 1.0)*0.5*lambda_fixed*phi.get_value(q), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble cell term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, 1),
                                                               phi_pres_fixed(data, 1);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_src(data, 0);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres_fixed.reinit(cell);
      phi_pres_fixed.gather_evaluate(pres_fixed, true, false);
      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, true, false);
      phi.reinit(cell);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& pres_fixed = phi_pres_fixed.get_value(q);

        phi.submit_gradient(-coeff*dt*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_fixed*phi_src.get_value(q), q);
      }
      phi.integrate_scatter(false, true, dst);
    }
  }


  // Assemble face term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_face_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_p(data, true, 1),
                                                                   phi_m(data, false, 1),
                                                                   phi_pres_fixed_p(data, true, 1),
                                                                   phi_pres_fixed_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_src_p(data, true, 0),
                                                                     phi_src_m(data, false, 0);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_pres_fixed_p.reinit(face);
      phi_pres_fixed_p.gather_evaluate(pres_fixed, true, false);
      phi_pres_fixed_m.reinit(face);
      phi_pres_fixed_m.gather_evaluate(pres_fixed, true, false);
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, true, false);
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, true, false);
      phi_p.reinit(face);
      phi_m.reinit(face);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus            = phi_p.get_normal_vector(q);

        const auto& pres_fixed_p      = phi_pres_fixed_p.get_value(q);
        const auto& pres_fixed_m      = phi_pres_fixed_m.get_value(q);
        const auto& avg_flux_enthalpy = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                          (pres_fixed_p*phi_src_p.get_value(q) + pres_fixed_m*phi_src_m.get_value(q));

        phi_p.submit_value(coeff*dt*scalar_product(avg_flux_enthalpy, n_plus), q);
        phi_m.submit_value(-coeff*dt*scalar_product(avg_flux_enthalpy, n_plus), q);
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }


  // Assemble boundary term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_boundary_term_enthalpy(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi(data, true, 1),
                                                                   phi_pres_fixed(data, true, 1);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_src(data, true, 0);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_pres_fixed.reinit(face);
      phi_pres_fixed.gather_evaluate(pres_fixed, true, false);
      phi_src.reinit(face);
      phi_src.gather_evaluate(src, true, false);
      phi.reinit(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus            = phi.get_normal_vector(q);

        const auto& pres_fixed        = phi_pres_fixed.get_value(q);
        const auto& avg_flux_enthalpy = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_fixed*phi_src.get_value(q);

        phi.submit_value(coeff*dt*scalar_product(avg_flux_enthalpy, n_plus), q);
      }
      phi.integrate_scatter(true, false, dst);
    }
  }



  // Assemble cell term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_src(data, 1);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, true, false);
      phi.reinit(cell);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_divergence(-coeff*dt/(Ma*Ma)*phi_src.get_value(q), q);
      phi.integrate_scatter(false, true, dst);
    }
  }


  // Assemble face term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                     phi_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_src_p(data, true, 1),
                                                                   phi_src_m(data, false, 1);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, true, false);
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, true, false);
      phi_p.reinit(face);
      phi_m.reinit(face);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus     = phi_p.get_normal_vector(q);

        const auto& avg_term   = 0.5*(phi_src_p.get_value(q) + phi_src_m.get_value(q));

        phi_p.submit_value(coeff*dt/(Ma*Ma)*avg_term*n_plus, q);
        phi_m.submit_value(-coeff*dt/(Ma*Ma)*avg_term*n_plus, q);
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }


  // Assemble boundary term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_src(data, true, 1);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_src.reinit(face);
      phi_src.gather_evaluate(src, true, true);
      phi.reinit(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus   = phi.get_normal_vector(q);

        const auto& avg_term = 0.5*phi_src.get_value(q);

        phi.submit_value(coeff*dt/(Ma*Ma)*avg_term*n_plus, q);
      }
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    Tensor<1, dim, VectorizedArray<Number>> e_k;
    for(unsigned int d = 0; d < dim - 1; ++d)
      e_k[d] = make_vectorized_array<Number>(0.0);
    e_k[dim - 1] = make_vectorized_array<Number>(1.0);

    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                   phi_u_old(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, 2);
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, true);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, true);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);
          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = pres_old;

          phi.submit_value(rho_old*u_old - 0.0*a21*dt/(Fr*Fr)*rho_old*e_k, q);
          phi.submit_gradient(a21*dt*rho_old*tensor_product_u_n + a21_tilde*dt/(Ma*Ma)*p_n_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                   phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old(data, 1),
                                                                 phi_pres_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, 2),
                                                                     phi_rho_tmp_2(data, 2);
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, true);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, true);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, true);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, true);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);
        phi.reinit(cell);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);
          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = pres_old;

          const auto& rho_tmp_2              = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2                = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2             = phi_pres_tmp_2.get_value(q);
          const auto& tensor_product_u_tmp_2 = outer_product(u_tmp_2, u_tmp_2);
          auto p_tmp_2_times_identity        = tensor_product_u_tmp_2;
          p_tmp_2_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_tmp_2_times_identity[d][d] = pres_tmp_2;

          phi.submit_value(rho_old*u_old - 0.0*a31*dt/(Fr*Fr)*rho_old*e_k - 0.0*a32*dt/(Fr*Fr)*rho_tmp_2*e_k, q);
          phi.submit_gradient(a31*dt*rho_old*tensor_product_u_n + a31_tilde*dt/(Ma*Ma)*p_n_times_identity +
                              a32*dt*rho_tmp_2*tensor_product_u_tmp_2 + a32_tilde*dt/(Ma*Ma)*p_tmp_2_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old_p(data, true, 1),
                                                                     phi_pres_old_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old_p(data, true, 2),
                                                                         phi_rho_old_m(data, false, 2);
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                 = phi_p.get_normal_vector(q);

          const auto& rho_old_p              = phi_rho_old_p.get_value(q);
          const auto& rho_old_m              = phi_rho_old_m.get_value(q);
          const auto& u_old_p                = phi_u_old_p.get_value(q);
          const auto& u_old_m                = phi_u_old_m.get_value(q);
          const auto& pres_old_p             = phi_pres_old_p.get_value(q);
          const auto& pres_old_m             = phi_pres_old_m.get_value(q);
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(rho_old_p*u_old_p, u_old_p) +
                                                    outer_product(rho_old_m*u_old_m, u_old_m));
          const auto& avg_pres_old           = 0.5*(pres_old_p + pres_old_m);
          const auto& jump_rhou_old          = rho_old_p*u_old_p - rho_old_m*u_old_m;
          const auto& lambda_old             = std::max(scalar_product(u_old_p, u_old_p) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_old_p/rho_old_p),
                                                        scalar_product(u_old_m, u_old_m) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_old_m/rho_old_m));

          phi_p.submit_value(-a21*dt*avg_tensor_product_u_n*n_plus
                             -a21_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                             -a21*dt*0.5*lambda_old*jump_rhou_old, q);
          phi_m.submit_value(a21*dt*avg_tensor_product_u_n*n_plus +
                             a21_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus +
                             a21*dt*0.5*lambda_old*jump_rhou_old, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old_p(data, true, 1),
                                                                     phi_pres_old_m(data, false, 1),
                                                                     phi_pres_tmp_2_p(data, true, 1),
                                                                     phi_pres_tmp_2_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old_p(data, true, 2),
                                                                         phi_rho_old_m(data, false, 2),
                                                                         phi_rho_tmp_2_p(data, true, 2),
                                                                         phi_rho_tmp_2_m(data, false, 2);
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], true, false);
        phi_p.reinit(face);
        phi_m.reinit(face);
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                 = phi_p.get_normal_vector(q);

          const auto& rho_old_p              = phi_rho_old_p.get_value(q);
          const auto& rho_old_m              = phi_rho_old_m.get_value(q);
          const auto& u_old_p                = phi_u_old_p.get_value(q);
          const auto& u_old_m                = phi_u_old_m.get_value(q);
          const auto& pres_old_p             = phi_pres_old_p.get_value(q);
          const auto& pres_old_m             = phi_pres_old_m.get_value(q);
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(rho_old_p*u_old_p, u_old_p) +
                                                    outer_product(rho_old_m*u_old_m, u_old_m));
          const auto& avg_pres_old           = 0.5*(pres_old_p + pres_old_m);
          const auto& jump_rhou_old          = rho_old_p*u_old_p - rho_old_m*u_old_m;
          const auto& lambda_old             = std::max(scalar_product(u_old_p, u_old_p) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_old_p/rho_old_p),
                                                        scalar_product(u_old_m, u_old_m) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_old_m/rho_old_m));

          const auto& rho_tmp_2_p                = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m                = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p                  = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m                  = phi_u_tmp_2_m.get_value(q);
          const auto& pres_tmp_2_p               = phi_pres_tmp_2_p.get_value(q);
          const auto& pres_tmp_2_m               = phi_pres_tmp_2_m.get_value(q);
          const auto& avg_tensor_product_u_tmp_2 = 0.5*(outer_product(rho_tmp_2_p*u_tmp_2_p, u_tmp_2_p) +
                                                        outer_product(rho_tmp_2_m*u_tmp_2_m, u_tmp_2_m));
          const auto& avg_pres_tmp_2             = 0.5*(pres_tmp_2_p + pres_tmp_2_m);
          const auto& jump_rhou_tmp_2            = rho_tmp_2_p*u_tmp_2_p - rho_tmp_2_m*u_tmp_2_m;
          const auto& lambda_tmp_2               = std::max(scalar_product(u_tmp_2_p, u_tmp_2_p) +
                                                            std::sqrt(EquationData::Cp_Cv*pres_tmp_2_p/rho_tmp_2_p),
                                                            scalar_product(u_tmp_2_m, u_tmp_2_m) +
                                                            std::sqrt(EquationData::Cp_Cv*pres_tmp_2_m/rho_tmp_2_m));

          phi_p.submit_value(-a31*dt*avg_tensor_product_u_n*n_plus
                             -a31_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                             -a31*dt*0.5*lambda_old*jump_rhou_old
                             -a32*dt*avg_tensor_product_u_tmp_2*n_plus
                             -a32_tilde*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus
                             -a32*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2, q);
          phi_m.submit_value(a31*dt*avg_tensor_product_u_n*n_plus +
                             a31_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus +
                             a31*dt*0.5*lambda_old*jump_rhou_old +
                             a32*dt*avg_tensor_product_u_tmp_2*n_plus +
                             a32_tilde*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus +
                             a32*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                       phi_u_old(data, true, 0),
                                                                       phi_u_fixed(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old(data, true, 1),
                                                                     phi_pres_fixed(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, true, 2),
                                                                         phi_rho_tmp_2(data, true, 2);

      auto pres_boundary_tmp_2 = pres_boundary;
      auto rho_boundary_tmp_2  = rho_boundary;
      auto u_boundary_tmp_2    = u_boundary;
      pres_boundary_tmp_2.advance_time(gamma*dt);
      rho_boundary_tmp_2.advance_time(gamma*dt);
      u_boundary_tmp_2.advance_time(gamma*dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_fixed.reinit(face);
        phi_u_fixed.gather_evaluate(src[4], true, false);
        phi_pres_fixed.reinit(face);
        phi_pres_fixed.gather_evaluate(src[5], true, false);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_old_D        = VectorizedArray<Number>();
          auto        pres_old_D       = VectorizedArray<Number>();
          auto        u_old_D          = Tensor<1, dim, VectorizedArray<Number>>();
          auto        rho_tmp_2_D      = VectorizedArray<Number>();
          auto        pres_tmp_2_D     = VectorizedArray<Number>();
          auto        u_tmp_2_D        = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_old_D[v]   = pres_boundary.value(point);
            pres_tmp_2_D[v] = pres_boundary_tmp_2.value(point);
            rho_old_D[v]    = rho_boundary.value(point);
            rho_tmp_2_D[v]  = rho_boundary_tmp_2.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v]   = u_boundary.value(point, d);
              u_tmp_2_D[d][v] = u_boundary_tmp_2.value(point, d);
            }
          }

          const auto& rho_old                = phi_rho_old.get_value(q);
          const auto& u_old                  = phi_u_old.get_value(q);
          const auto& pres_old               = phi_pres_old.get_value(q);
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(rho_old*u_old, u_old) +
                                                    outer_product(rho_old_D*u_old_D, u_old_D));
          const auto& avg_pres_old           = 0.5*(pres_old + pres_old_D);
          const auto& jump_rhou_old          = rho_old*u_old - rho_old_D*u_old_D;
          const auto& lambda_old             = std::max(scalar_product(u_old, u_old) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_old/rho_old),
                                                        scalar_product(u_old_D, u_old_D) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));

          const auto& rho_tmp_2              = phi_rho_tmp_2.get_value(q);
          const auto& u_fixed                = phi_u_fixed.get_value(q);
          const auto& pres_fixed             = phi_pres_fixed.get_value(q);
          const auto& lambda_fixed           = std::max(scalar_product(u_fixed, u_fixed) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_fixed/rho_tmp_2),
                                                        scalar_product(u_tmp_2_D, u_tmp_2_D) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));

          phi.submit_value(-a21*dt*avg_tensor_product_u_n*n_plus
                           -a21_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                           -a21*dt*0.5*lambda_old*jump_rhou_old
                           -a22_tilde*dt/(Ma*Ma)*0.5*pres_tmp_2_D*n_plus
                           +a22_tilde*dt*0.5*lambda_fixed*rho_tmp_2_D*u_tmp_2_D, q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                       phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0),
                                                                       phi_u_fixed(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_old(data, true, 1),
                                                                     phi_pres_tmp_2(data, true, 1),
                                                                     phi_pres_fixed(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_old(data, true, 2),
                                                                         phi_rho_tmp_2(data, true, 2),
                                                                         phi_rho_curr(data, true, 2);

      auto pres_boundary_tmp_2 = pres_boundary;
      auto rho_boundary_tmp_2  = rho_boundary;
      auto u_boundary_tmp_2    = u_boundary;
      pres_boundary_tmp_2.advance_time(gamma*dt);
      rho_boundary_tmp_2.advance_time(gamma*dt);
      u_boundary_tmp_2.advance_time(gamma*dt);
      auto pres_boundary_curr = pres_boundary;
      auto rho_boundary_curr  = rho_boundary;
      auto u_boundary_curr    = u_boundary;
      pres_boundary_curr.advance_time(dt);
      rho_boundary_curr.advance_time(dt);
      u_boundary_curr.advance_time(dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);
        phi_rho_curr.reinit(face);
        phi_rho_curr.gather_evaluate(src[6], true, false);
        phi_u_fixed.reinit(face);
        phi_u_fixed.gather_evaluate(src[7], true, false);
        phi_pres_fixed.reinit(face);
        phi_pres_fixed.gather_evaluate(src[8], true, false);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_old_D        = VectorizedArray<Number>();
          auto        pres_old_D       = VectorizedArray<Number>();
          auto        u_old_D          = Tensor<1, dim, VectorizedArray<Number>>();
          auto        rho_tmp_2_D      = VectorizedArray<Number>();
          auto        pres_tmp_2_D     = VectorizedArray<Number>();
          auto        u_tmp_2_D        = Tensor<1, dim, VectorizedArray<Number>>();
          auto        rho_curr_D       = VectorizedArray<Number>();
          auto        pres_curr_D      = VectorizedArray<Number>();
          auto        u_curr_D         = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_old_D[v]   = pres_boundary.value(point);
            pres_tmp_2_D[v] = pres_boundary_tmp_2.value(point);
            pres_curr_D[v]  = pres_boundary_curr.value(point);
            rho_old_D[v]    = rho_boundary.value(point);
            rho_tmp_2_D[v]  = rho_boundary_tmp_2.value(point);
            rho_curr_D[v]   = rho_boundary_curr.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v]   = u_boundary.value(point, d);
              u_tmp_2_D[d][v] = u_boundary_tmp_2.value(point, d);
              u_curr_D[d][v]  = u_boundary_curr.value(point, d);
            }
          }

          const auto& rho_old                = phi_rho_old.get_value(q);
          const auto& u_old                  = phi_u_old.get_value(q);
          const auto& pres_old               = phi_pres_old.get_value(q);
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(rho_old*u_old, u_old) +
                                                    outer_product(rho_old_D*u_old_D, u_old_D));
          const auto& avg_pres_old           = 0.5*(pres_old + pres_old_D);
          const auto& jump_rhou_old          = rho_old*u_old - rho_old_D*u_old_D;
          const auto& lambda_old             = std::max(scalar_product(u_old, u_old) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_old/rho_old),
                                                        scalar_product(u_old_D, u_old_D) +
                                                        std::sqrt(EquationData::Cp_Cv*pres_old_D/rho_old_D));

          const auto& rho_tmp_2                  = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2                    = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2                 = phi_pres_tmp_2.get_value(q);
          const auto& avg_tensor_product_u_tmp_2 = 0.5*(outer_product(rho_tmp_2*u_tmp_2, u_tmp_2) +
                                                        outer_product(rho_tmp_2_D*u_tmp_2_D, u_tmp_2_D));
          const auto& avg_pres_tmp_2             = 0.5*(pres_tmp_2 + pres_tmp_2_D);
          const auto& jump_rhou_tmp_2            = rho_tmp_2*u_tmp_2 - rho_tmp_2_D*u_tmp_2_D;
          const auto& lambda_tmp_2               = std::max(scalar_product(u_tmp_2, u_tmp_2) +
                                                            std::sqrt(EquationData::Cp_Cv*pres_tmp_2/rho_tmp_2),
                                                            scalar_product(u_tmp_2_D, u_tmp_2_D) +
                                                            std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));

          const auto& rho_curr     = phi_rho_curr.get_value(q);
          const auto& u_fixed      = phi_u_fixed.get_value(q);
          const auto& pres_fixed   = phi_pres_fixed.get_value(q);
          const auto& lambda_fixed = std::max(scalar_product(u_fixed, u_fixed) +
                                              std::sqrt(EquationData::Cp_Cv*pres_fixed/rho_curr),
                                              scalar_product(u_curr_D, u_curr_D) +
                                              std::sqrt(EquationData::Cp_Cv*pres_curr_D/rho_curr_D));

          phi.submit_value(-a31*dt*avg_tensor_product_u_n*n_plus
                           -a31_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                           -a31*dt*0.5*lambda_old*jump_rhou_old
                           -a32*dt*avg_tensor_product_u_tmp_2*n_plus
                           -a32_tilde*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus
                           -a32*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2
                           -a33_tilde*dt/(Ma*Ma)*0.5*pres_curr_D*n_plus
                           +a33_tilde*dt*0.5*lambda_fixed*rho_curr_D*u_curr_D, q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_velocity_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();
    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_velocity_update,
                     &HYPERBOLICOperator::assemble_rhs_face_term_velocity_update,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_velocity_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_for_fixed(data, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, true, false);
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi_rho_for_fixed.get_value(q)*phi.get_value(q), q);
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble face term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_face_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                     phi_m(data, false, 0),
                                                                     phi_u_fixed_p(data, true, 0),
                                                                     phi_u_fixed_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_fixed_p(data, true, 1),
                                                                   phi_pres_fixed_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_for_fixed_p(data, true, 2),
                                                                       phi_rho_for_fixed_m(data, false, 2);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_u_fixed_p.reinit(face);
      phi_u_fixed_p.gather_evaluate(u_fixed, true, false);
      phi_u_fixed_m.reinit(face);
      phi_u_fixed_m.gather_evaluate(u_fixed, true, false);
      phi_rho_for_fixed_p.reinit(face);
      phi_rho_for_fixed_p.gather_evaluate(rho_for_fixed, true, false);
      phi_rho_for_fixed_m.reinit(face);
      phi_rho_for_fixed_m.gather_evaluate(rho_for_fixed, true, false);
      phi_pres_fixed_p.reinit(face);
      phi_pres_fixed_p.gather_evaluate(pres_fixed, true, false);
      phi_pres_fixed_m.reinit(face);
      phi_pres_fixed_m.gather_evaluate(pres_fixed, true, false);
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, true, false);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, true, false);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& u_fixed_p       = phi_u_fixed_p.get_value(q);
        const auto& u_fixed_m       = phi_u_fixed_m.get_value(q);
        const auto& pres_fixed_p    = phi_pres_fixed_p.get_value(q);
        const auto& pres_fixed_m    = phi_pres_fixed_m.get_value(q);
        const auto& rho_for_fixed_p = phi_rho_for_fixed_p.get_value(q);
        const auto& rho_for_fixed_m = phi_rho_for_fixed_m.get_value(q);
        const auto& lambda_fixed    = std::max(scalar_product(u_fixed_p, u_fixed_p) +
                                               std::sqrt(EquationData::Cp_Cv*pres_fixed_p/rho_for_fixed_p),
                                               scalar_product(u_fixed_m, u_fixed_m) +
                                               std::sqrt(EquationData::Cp_Cv*pres_fixed_m/rho_for_fixed_m));
        const auto& jump_term       = (rho_for_fixed_p*phi_p.get_value(q) -
                                       rho_for_fixed_m*phi_m.get_value(q));

        phi_p.submit_value(coeff*dt*0.5*lambda_fixed*jump_term, q);
        phi_m.submit_value(-coeff*dt*0.5*lambda_fixed*jump_term, q);
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }


  // Assemble boundary term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_boundary_term_velocity_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                       phi_u_fixed(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_fixed(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_for_fixed(data, true, 2);

      auto pres_boundary_tmp_2 = pres_boundary;
      auto rho_boundary_tmp_2  = rho_boundary;
      auto u_boundary_tmp_2    = u_boundary;
      pres_boundary_tmp_2.advance_time(gamma*dt);
      rho_boundary_tmp_2.advance_time(gamma*dt);
      u_boundary_tmp_2.advance_time(gamma*dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_u_fixed.reinit(face);
        phi_u_fixed.gather_evaluate(u_fixed, true, false);
        phi_rho_for_fixed.reinit(face);
        phi_rho_for_fixed.gather_evaluate(rho_for_fixed, true, false);
        phi_pres_fixed.reinit(face);
        phi_pres_fixed.gather_evaluate(pres_fixed, true, false);
        phi.reinit(face);
        phi.gather_evaluate(src, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_tmp_2_D      = VectorizedArray<Number>();
          auto        pres_tmp_2_D     = VectorizedArray<Number>();
          auto        u_tmp_2_D        = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_tmp_2_D[v] = pres_boundary_tmp_2.value(point);
            rho_tmp_2_D[v]  = rho_boundary_tmp_2.value(point);
            for(unsigned int d = 0; d < dim; ++d)
              u_tmp_2_D[d][v] = u_boundary_tmp_2.value(point, d);
          }

          const auto& u_fixed       = phi_u_fixed.get_value(q);
          const auto& pres_fixed    = phi_pres_fixed.get_value(q);
          const auto& rho_for_fixed = phi_rho_for_fixed.get_value(q);
          const auto& lambda_fixed  = std::max(scalar_product(u_fixed, u_fixed) +
                                               std::sqrt(EquationData::Cp_Cv*pres_fixed/rho_for_fixed),
                                               scalar_product(u_tmp_2_D, u_tmp_2_D) +
                                               std::sqrt(EquationData::Cp_Cv*pres_tmp_2_D/rho_tmp_2_D));

          phi.submit_value(a22_tilde*dt*0.5*lambda_fixed*rho_for_fixed*phi.get_value(q), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                       phi_u_fixed(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres_fixed(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho_for_fixed(data, true, 2);

      auto pres_boundary_curr = pres_boundary;
      auto rho_boundary_curr  = rho_boundary;
      auto u_boundary_curr    = u_boundary;
      pres_boundary_curr.advance_time(dt);
      rho_boundary_curr.advance_time(dt);
      u_boundary_curr.advance_time(dt);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_u_fixed.reinit(face);
        phi_u_fixed.gather_evaluate(u_fixed, true, false);
        phi_rho_for_fixed.reinit(face);
        phi_rho_for_fixed.gather_evaluate(rho_for_fixed, true, false);
        phi_pres_fixed.reinit(face);
        phi_pres_fixed.gather_evaluate(pres_fixed, true, false);
        phi.reinit(face);
        phi.gather_evaluate(src, true, false);
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& point_vectorized = phi.quadrature_point(q);
          auto        rho_curr_D      = VectorizedArray<Number>();
          auto        pres_curr_D     = VectorizedArray<Number>();
          auto        u_curr_D        = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d)
              point[d] = point_vectorized[d][v];
            pres_curr_D[v] = pres_boundary_curr.value(point);
            rho_curr_D[v]  = rho_boundary_curr.value(point);
            for(unsigned int d = 0; d < dim; ++d)
              u_curr_D[d][v] = u_boundary_curr.value(point, d);
          }

          const auto& u_fixed       = phi_u_fixed.get_value(q);
          const auto& pres_fixed    = phi_pres_fixed.get_value(q);
          const auto& rho_for_fixed = phi_rho_for_fixed.get_value(q);
          const auto& lambda_fixed  = std::max(scalar_product(u_fixed, u_fixed) +
                                               std::sqrt(EquationData::Cp_Cv*pres_fixed/rho_for_fixed),
                                               scalar_product(u_curr_D, u_curr_D) +
                                               std::sqrt(EquationData::Cp_Cv*pres_curr_D/rho_curr_D));

          phi.submit_value(a33_tilde*dt*0.5*lambda_fixed*rho_for_fixed*phi.get_value(q), q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(NS_stage, 4);
    Assert(NS_stage > 0, ExcInternalError());
    if(NS_stage == 1) {
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_rho_projection,
                            this, dst, src, false);
    }
    else if(NS_stage == 2){
      this->data->loop(&HYPERBOLICOperator::assemble_cell_term_internal_energy,
                       &HYPERBOLICOperator::assemble_face_term_internal_energy,
                       &HYPERBOLICOperator::assemble_boundary_term_internal_energy,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);

      Vec tmp_1;
      this->data->initialize_dof_vector(tmp_1, 0);
      this->vmult_pressure(tmp_1, src);

      NS_stage = 3;
      Vec tmp_2;
      this->data->initialize_dof_vector(tmp_2, 0);
      tmp_2 = 0;
      const std::vector<unsigned int> tmp_reinit = {0};
      auto* tmp_matrix = const_cast<HYPERBOLICOperator*>(this);
      tmp_matrix->initialize(tmp_matrix->get_matrix_free(), tmp_reinit, tmp_reinit);
      SolverControl solver_control(10000, 1e-12*tmp_1.l2_norm());
      SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);
      gmres.solve(*tmp_matrix, tmp_2, tmp_1, PreconditionIdentity());

      Vec tmp_3;
      this->data->initialize_dof_vector(tmp_3, 1);
      this->vmult_enthalpy(tmp_3, tmp_2);

      dst.add(-1.0, tmp_3);
      NS_stage = 2;
      const std::vector<unsigned int> tmp = {1};
      tmp_matrix->initialize(tmp_matrix->get_matrix_free(), tmp, tmp);
    }
    else {
      this->data->loop(&HYPERBOLICOperator::assemble_cell_term_velocity_update,
                       &HYPERBOLICOperator::assemble_face_term_velocity_update,
                       &HYPERBOLICOperator::assemble_boundary_term_velocity_update,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
  }


  // Application of pressure matrix
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_pressure(Vec& dst, const Vec& src) const {
    src.update_ghost_values();
    this->data->loop(&HYPERBOLICOperator::assemble_cell_term_pressure,
                     &HYPERBOLICOperator::assemble_face_term_pressure,
                     &HYPERBOLICOperator::assemble_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Application of enthalpy matrix
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_enthalpy(Vec& dst, const Vec& src) const {
    src.update_ghost_values();
    this->data->loop(&HYPERBOLICOperator::assemble_cell_term_enthalpy,
                     &HYPERBOLICOperator::assemble_face_term_enthalpy,
                     &HYPERBOLICOperator::assemble_boundary_term_enthalpy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Compute max celerity
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  Number HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                            n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  compute_max_celerity(const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();

    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u(*(this->data), 0);
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_T, 1, Number> phi_pres(*(this->data), 1);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_rho, 1, Number> phi_rho(*(this->data), 2);

    Number max_celerity = 0.0;

    for(unsigned int cell = 0; cell < this->data->n_cell_batches(); ++cell) {
      phi_rho.reinit(cell);
      phi_rho.gather_evaluate(src[0], true, false);
      phi_u.reinit(cell);
      phi_u.gather_evaluate(src[1], true, false);
      phi_pres.reinit(cell);
      phi_pres.gather_evaluate(src[2], true, false);

      VectorizedArray<Number> local_max = 0.;

      for(unsigned int q = 0; q < phi_u.n_q_points; ++q) {
        const auto& rho  = phi_rho.get_value(q);
        const auto& u    = phi_u.get_value(q);
        const auto& pres = phi_pres.get_value(q);

        VectorizedArray<Number> convective_speed = 0.;
        for(unsigned int d = 0; d < dim; ++d)
          convective_speed = std::max(convective_speed, std::abs(u[d]));
        const auto& speed_of_sound = std::sqrt(EquationData::Cp_Cv*pres/rho);

        local_max = std::max(local_max, convective_speed + speed_of_sound);
      }
      for(unsigned int v = 0; v < this->data->n_active_entries_per_cell_batch(cell); ++v)
        max_celerity = std::max(max_celerity, local_max[v]);
    }

    max_celerity = Utilities::MPI::max(max_celerity, MPI_COMM_WORLD);

    return max_celerity;
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
    LinearAlgebra::distributed::Vector<double> u_fixed;
    LinearAlgebra::distributed::Vector<double> rhs_u;

    LinearAlgebra::distributed::Vector<double> pres_old;
    LinearAlgebra::distributed::Vector<double> pres_tmp_2;
    LinearAlgebra::distributed::Vector<double> pres_fixed;
    LinearAlgebra::distributed::Vector<double> pres_fixed_old;
    LinearAlgebra::distributed::Vector<double> pres_tmp;
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

    void pressure_fixed_point();

    void update_velocity();

    void analyze_results();

    void output_results(const unsigned int step);

  private:
    EquationData::Density<dim>  rho_exact;
    EquationData::Velocity<dim> u_exact;
    EquationData::Pressure<dim> pres_exact;

    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    HYPERBOLICOperator<dim, EquationData::degree_rho, EquationData::degree_T, EquationData::degree_u,
                            EquationData::degree_rho + 1, EquationData::degree_T + 1,EquationData::degree_u + 1,
                            LinearAlgebra::distributed::Vector<double>, double> navier_stokes_matrix;

    MGLevelObject<HYPERBOLICOperator<dim, EquationData::degree_rho, EquationData::degree_T, EquationData::degree_u,
                                          EquationData::degree_rho + 1, EquationData::degree_T + 1, EquationData::degree_u + 1,
                                          LinearAlgebra::distributed::Vector<double>, double>> mg_matrices;

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
                   H1_error_per_cell_pres,
                   L2_error_per_cell_pres,
                   H1_rel_error_per_cell_pres,
                   L2_rel_error_per_cell_pres,
                   Linfty_error_per_cell_vel,
                   Linfty_error_per_cell_pres;

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
    quadrature_density(EquationData::degree_rho + 2),
    quadrature_velocity(EquationData::degree_u + 2),
    quadrature_temperature(EquationData::degree_T + 2),
    rho_exact(data.initial_time),
    u_exact(data.initial_time),
    pres_exact(data.initial_time),
    navier_stokes_matrix(data),
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
    for(unsigned int d = 1; d < dim; ++d)
      lower_left[d] = -5;

    Point<dim> upper_right;
    upper_right[0] = 10;
    for(unsigned int d = 1; d < dim; ++d)
      upper_right[d] = 5;

    GridGenerator::hyper_rectangle(triangulation, lower_left, upper_right);
    triangulation.refine_global(2);
    triangulation.refine_global(n_refines);

    //dt = CFL*10.0/(std::pow(2, n_refines + 2))/(EquationData::degree_rho);
    //navier_stokes_matrix.set_dt(dt);
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
    quadratures.push_back(QGauss<1>(EquationData::degree_u + 1));
    quadratures.push_back(QGauss<1>(EquationData::degree_T + 1));
    quadratures.push_back(QGauss<1>(EquationData::degree_rho + 1));

    matrix_free_storage->reinit(dof_handlers, constraints, quadratures, additional_data);

    matrix_free_storage->initialize_dof_vector(u_old, 0);
    matrix_free_storage->initialize_dof_vector(u_tmp_2, 0);
    matrix_free_storage->initialize_dof_vector(u_curr, 0);
    matrix_free_storage->initialize_dof_vector(u_fixed, 0);
    matrix_free_storage->initialize_dof_vector(rhs_u, 0);

    matrix_free_storage->initialize_dof_vector(pres_old, 1);
    matrix_free_storage->initialize_dof_vector(pres_tmp_2, 1);
    matrix_free_storage->initialize_dof_vector(pres_fixed, 1);
    matrix_free_storage->initialize_dof_vector(pres_fixed_old, 1);
    matrix_free_storage->initialize_dof_vector(pres_tmp, 1);
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
    H1_error_per_cell_pres.reinit(error_per_cell_tmp);
    L2_error_per_cell_pres.reinit(error_per_cell_tmp);
    H1_rel_error_per_cell_pres.reinit(error_per_cell_tmp);
    L2_rel_error_per_cell_pres.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_vel.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_pres.reinit(error_per_cell_tmp);
  }


  // @sect4{ <code>NavierStokesProjection::initialize</code> }

  // This method loads the initial data
  //
  template<int dim>
  void NavierStokesProjection<dim>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize state");

    VectorTools::interpolate(dof_handler_density, rho_exact, rho_old);
    VectorTools::interpolate(dof_handler_velocity, u_exact, u_old);
    VectorTools::interpolate(dof_handler_temperature, pres_exact, pres_old);
  }


  // @sect4{<code>NavierStokesProjection::update_density</code>}

  // This implements the update of the density for the hyperbolic part
  //
  template<int dim>
  void NavierStokesProjection<dim>::update_density() {
    TimerOutput::Scope t(time_table, "Update density");

    const std::vector<unsigned int> tmp = {2};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

    navier_stokes_matrix.set_NS_stage(1);

    if(HYPERBOLIC_stage == 1)
      navier_stokes_matrix.vmult_rhs_rho_projection(rhs_rho, {rho_old, u_old, pres_old});
    else
      navier_stokes_matrix.vmult_rhs_rho_projection(rhs_rho, {rho_old, u_old, pres_old,
                                                              rho_tmp_2, u_tmp_2, pres_tmp_2});

    SolverControl solver_control(vel_max_its, vel_eps*rhs_rho.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    if(HYPERBOLIC_stage == 1) {
      rho_tmp_2 = rho_old;
      cg.solve(navier_stokes_matrix, rho_tmp_2, rhs_rho, PreconditionIdentity());
    }
    else {
      rho_curr = rho_tmp_2;
      cg.solve(navier_stokes_matrix, rho_curr, rhs_rho, PreconditionIdentity());
    }
  }


  // @sect4{<code>NavierStokesProjection::pressure_fixed_point</code>}

  // This implements a step of the fixed point procedure for the computation of the pressure in the hyperbolic part
  //
  template<int dim>
  void NavierStokesProjection<dim>::pressure_fixed_point() {
    TimerOutput::Scope t(time_table, "Fixed point pressure");

    const std::vector<unsigned int> tmp = {1};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

    navier_stokes_matrix.set_NS_stage(2);

    navier_stokes_matrix.set_pres_fixed(pres_fixed_old);
    navier_stokes_matrix.set_u_fixed(u_fixed);
    if(HYPERBOLIC_stage == 1) {
      navier_stokes_matrix.vmult_rhs_pressure(rhs_pres, {rho_old, u_old, pres_old,
                                                         rho_tmp_2, u_fixed, pres_fixed_old});

      navier_stokes_matrix.vmult_rhs_velocity_update(rhs_u, {rho_old, u_old, pres_old,
                                                             rho_tmp_2, u_fixed, pres_fixed_old});
    }
    else {
      navier_stokes_matrix.vmult_rhs_pressure(rhs_pres, {rho_old, u_old, pres_old,
                                                         rho_tmp_2, u_tmp_2, pres_tmp_2,
                                                         rho_curr, u_fixed, pres_fixed_old});

      navier_stokes_matrix.vmult_rhs_velocity_update(rhs_u, {rho_old, u_old, pres_old,
                                                             rho_tmp_2, u_tmp_2, pres_tmp_2,
                                                             rho_curr, u_fixed, pres_fixed_old});
    }

    SolverControl solver_control_schur(vel_max_its, 1e-12*rhs_u.l2_norm());
    SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres_schur(solver_control_schur);
    const std::vector<unsigned int> tmp_reinit = {0};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp_reinit, tmp_reinit);
    LinearAlgebra::distributed::Vector<double> tmp_1;
    matrix_free_storage->initialize_dof_vector(tmp_1, 0);
    tmp_1 = 0;
    navier_stokes_matrix.set_NS_stage(3);
    gmres_schur.solve(navier_stokes_matrix, tmp_1, rhs_u, PreconditionIdentity());

    LinearAlgebra::distributed::Vector<double> tmp_2;
    matrix_free_storage->initialize_dof_vector(tmp_2, 1);
    navier_stokes_matrix.vmult_enthalpy(tmp_2, tmp_1);

    rhs_pres.add(-1.0, tmp_2);

    navier_stokes_matrix.set_NS_stage(2);
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

    SolverControl solver_control(vel_max_its, vel_eps*rhs_pres.l2_norm());
    SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);

    pres_fixed.equ(1.0, pres_fixed_old);
    gmres.solve(navier_stokes_matrix, pres_fixed, rhs_pres, PreconditionIdentity());
  }


  // @sect4{<code>NavierStokesProjection::update_velocity</code>}

  // This implements the velocity update in the fixed point procedure for the computation of the pressure in the hyperbolic part
  //
  template<int dim>
  void NavierStokesProjection<dim>::update_velocity() {
    TimerOutput::Scope t(time_table, "Update velocity");

    const std::vector<unsigned int> tmp = {0};
    navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

    navier_stokes_matrix.set_NS_stage(3);

    if(HYPERBOLIC_stage == 1) {
      LinearAlgebra::distributed::Vector<double> tmp_1;
      matrix_free_storage->initialize_dof_vector(tmp_1, 0);
      navier_stokes_matrix.vmult_pressure(tmp_1, pres_fixed);
      rhs_u.add(-1.0, tmp_1);
    }
    else {
      navier_stokes_matrix.vmult_rhs_velocity_update(rhs_u, {rho_old, u_old, pres_old,
                                                             rho_tmp_2, u_tmp_2, pres_tmp_2,
                                                             rho_curr, u_fixed, pres_fixed_old});

      LinearAlgebra::distributed::Vector<double> tmp_1;
      matrix_free_storage->initialize_dof_vector(tmp_1, 0);
      navier_stokes_matrix.vmult_pressure(tmp_1, pres_fixed);
      rhs_u.add(-1.0, tmp_1);
    }

    SolverControl solver_control(vel_max_its, vel_eps*rhs_u.l2_norm());
    SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);

    gmres.solve(navier_stokes_matrix, u_fixed, rhs_u, PreconditionIdentity());
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
    pres_old.update_ghost_values();
    data_out.add_data_vector(dof_handler_density, pres_old, "p", {DataComponentInterpretation::component_is_scalar});

    VectorTools::interpolate(dof_handler_density, rho_exact, rho_tmp_2);
    rho_tmp_2.add(-1.0, rho_old);
    rho_tmp_2.update_ghost_values();
    data_out.add_data_vector(dof_handler_density, rho_tmp_2,
                             "Nodal_error_density", {DataComponentInterpretation::component_is_scalar});
    VectorTools::interpolate(dof_handler_velocity, u_exact, u_tmp_2);
    u_tmp_2.add(-1.0, u_old);
    std::vector<std::string> velocity_names_error(dim, "Nodal_error_velocity");
    u_tmp_2.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_tmp_2, velocity_names_error, component_interpretation_velocity);
    VectorTools::interpolate(dof_handler_temperature, pres_exact, pres_tmp_2);
    pres_tmp_2.add(-1.0, pres_old);
    pres_tmp_2.update_ghost_values();
    data_out.add_data_vector(dof_handler_temperature, pres_tmp_2,
                             "Nodal_error_pressure", {DataComponentInterpretation::component_is_scalar});

    data_out.build_patches();
    const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
  }


  // Since we have solved a problem with analytic solution, we want to verify
  // the correctness of our implementation by computing the errors of the
  // numerical result against the analytic solution.
  //
  template <int dim>
  void NavierStokesProjection<dim>::analyze_results() {
    TimerOutput::Scope t(time_table, "Analysis results: computing errrors");

    u_tmp_2 = 0;

    VectorTools::integrate_difference(dof_handler_velocity, u_old, u_exact,
                                      L2_error_per_cell_vel, quadrature_velocity, VectorTools::L2_norm);
    const double error_vel_L2 = VectorTools::compute_global_error(triangulation, L2_error_per_cell_vel, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_velocity, u_tmp_2, u_exact,
                                      L2_rel_error_per_cell_vel, quadrature_velocity, VectorTools::L2_norm);
    const double L2_vel = VectorTools::compute_global_error(triangulation, L2_rel_error_per_cell_vel, VectorTools::L2_norm);
    const double error_rel_vel_L2 = error_vel_L2/L2_vel;
    pcout << "Verification via L2 error velocity:    "<< error_vel_L2 << std::endl;
    pcout << "Verification via L2 relative error velocity:    "<< error_rel_vel_L2 << std::endl;

    rho_tmp_2 = 0;

    VectorTools::integrate_difference(dof_handler_density, rho_old, rho_exact,
                                      L2_error_per_cell_rho, quadrature_density, VectorTools::L2_norm);
    const double error_rho_L2 = VectorTools::compute_global_error(triangulation, L2_error_per_cell_rho, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_density, rho_tmp_2, rho_exact,
                                      L2_rel_error_per_cell_rho, quadrature_density, VectorTools::L2_norm);
    const double L2_rho = VectorTools::compute_global_error(triangulation, L2_rel_error_per_cell_rho, VectorTools::L2_norm);
    const double error_rel_rho_L2 = error_rho_L2/L2_rho;
    pcout << "Verification via L2 error density:    "<< error_rho_L2 << std::endl;
    pcout << "Verification via L2 relative error density:    "<< error_rel_rho_L2 << std::endl;

    pres_tmp_2 = 0;

    VectorTools::integrate_difference(dof_handler_temperature, pres_old, pres_exact,
                                      L2_error_per_cell_pres, quadrature_temperature, VectorTools::L2_norm);
    const double error_pres_L2 = VectorTools::compute_global_error(triangulation, L2_error_per_cell_pres, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler_temperature, pres_tmp_2, pres_exact,
                                      L2_rel_error_per_cell_pres, quadrature_temperature, VectorTools::L2_norm);
    const double L2_pres = VectorTools::compute_global_error(triangulation, L2_rel_error_per_cell_pres, VectorTools::L2_norm);
    const double error_rel_pres_L2 = error_pres_L2/L2_pres;
    pcout << "Verification via L2 error pressure:    "<< error_pres_L2 << std::endl;
    pcout << "Verification via L2 relative error pressure:    "<< error_rel_pres_L2 << std::endl;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_error_vel << error_vel_L2 << std::endl;
      output_error_vel << error_rel_vel_L2 << std::endl;
      output_error_rho << error_rho_L2 << std::endl;
      output_error_rho << error_rel_rho_L2 << std::endl;
      output_error_pres << error_pres_L2 << std::endl;
      output_error_pres << error_rel_pres_L2 << std::endl;
    }
  }


  // The following function is used in determining the maximal velocity
  // in order to compute the CFL
  //
  template<int dim>
  double NavierStokesProjection<dim>::get_maximal_velocity() {
    VectorTools::integrate_difference(dof_handler_velocity, u_old, ZeroFunction<dim>(dim),
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

    analyze_results();
    output_results(0);
    double time = t_0;
    unsigned int n = 0;
    while(std::abs(T - time) > 1e-10) {
      time += dt;
      rho_exact.advance_time(gamma*dt);
      u_exact.advance_time(gamma*dt);
      pres_exact.advance_time(gamma*dt);
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;
      //--- First stage of HYPERBOLIC operator
      navier_stokes_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);
      verbose_cout << "  Update density stage 1" << std::endl;
      update_density();
      verbose_cout << "  Fixed point pressure stage 1" << std::endl;
      navier_stokes_matrix.set_rho_for_fixed(rho_tmp_2);
      pres_fixed_old.equ(1.0, pres_old);
      u_fixed.zero_out_ghosts();
      VectorTools::interpolate(dof_handler_velocity, u_exact, u_fixed);
      u_fixed.equ(1.0, u_old);
      for(unsigned int iter = 0; iter < 100; ++iter) {
        pressure_fixed_point();
        update_velocity();

        //Compute the relative error
        VectorTools::integrate_difference(dof_handler_temperature, pres_fixed, ZeroFunction<dim>(),
                                          Linfty_error_per_cell_pres, quadrature_temperature, VectorTools::Linfty_norm);
        const double den = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm);
        double error = 0.0;
        pres_tmp.equ(1.0, pres_fixed);
        pres_tmp.add(-1.0, pres_fixed_old);
        VectorTools::integrate_difference(dof_handler_temperature, pres_tmp, ZeroFunction<dim>(),
                                          Linfty_error_per_cell_pres, quadrature_temperature, VectorTools::Linfty_norm);
        error = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm)/den;
        if(error < 1e-6)
          break;

        pres_fixed_old.equ(1.0, pres_fixed);
      }
      pres_tmp_2.equ(1.0, pres_fixed);
      u_tmp_2.equ(1.0, u_fixed);
      HYPERBOLIC_stage = 2; //--- Flag to pass at second stage

      //--- Second stage of HYPERBOLIC operator
      navier_stokes_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);
      rho_exact.advance_time((1.0 - gamma)*dt);
      u_exact.advance_time((1.0 - gamma)*dt);
      pres_exact.advance_time((1.0 - gamma)*dt);
      verbose_cout << "  Update density stage 2" << std::endl;
      update_density();
      verbose_cout << "  Fixed point pressure stage 2" << std::endl;
      navier_stokes_matrix.set_rho_for_fixed(rho_curr);
      pres_fixed_old.equ(1.0, pres_tmp_2);
      u_fixed.equ(1.0, u_tmp_2);
      for(unsigned int iter = 0; iter < 100; ++iter) {
        pressure_fixed_point();
        update_velocity();

        //Compute the relative error
        VectorTools::integrate_difference(dof_handler_temperature, pres_fixed, ZeroFunction<dim>(),
                                          Linfty_error_per_cell_pres, quadrature_temperature, VectorTools::Linfty_norm);
        const double den = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm);
        double error = 0.0;
        pres_tmp.equ(1.0, pres_fixed);
        pres_tmp.add(-1.0, pres_fixed_old);
        VectorTools::integrate_difference(dof_handler_temperature, pres_tmp, ZeroFunction<dim>(),
                                          Linfty_error_per_cell_pres, quadrature_temperature, VectorTools::Linfty_norm);
        error = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm)/den;
        if(error < 1e-6)
          break;

        pres_fixed_old.equ(1.0, pres_fixed);
      }
      pres_old.equ(1.0, pres_fixed);
      u_curr.equ(1.0, u_fixed);
      HYPERBOLIC_stage = 1; //--- Flag to pass at first stage at next step

      //--- Update for next step
      navier_stokes_matrix.advance_rho_boundary_time(dt);
      navier_stokes_matrix.advance_pres_boundary_time(dt);
      navier_stokes_matrix.advance_u_boundary_time(dt);
      rho_old.equ(1.0, rho_curr);
      u_old.equ(1.0, u_curr);
      const double max_celerity = navier_stokes_matrix.compute_max_celerity({rho_old, u_old, pres_old});
      pcout<< "Maximal celerity = " << max_celerity << std::endl;
      pcout << "CFL = " << dt*max_celerity*std::pow((EquationData::degree_u), 1.5)*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;
      analyze_results();
      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution final" << std::endl;
        output_results(n);
      }
      //if(time > 0.1*T && get_maximal_difference() < 1e-7)
      //  break;
      if(T - time < dt && T - time > 1e-10) {
        dt = T - time;
        navier_stokes_matrix.set_dt(dt);
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
