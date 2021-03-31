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

#include <deal.II/grid/grid_out.h>

namespace Step35 {
  using namespace dealii;

  template<int dim>
  class NavierStokesProjection {
  public:
    NavierStokesProjection();

  protected:
    parallel::distributed::Triangulation<dim> triangulation;

    void create_triangulation_and_dofs();

  private:
    ConditionalOStream pcout;
  };


  // @sect4{ <code>NavierStokesProjection::NavierStokesProjection</code> }

  template<int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(): triangulation(MPI_COMM_WORLD),
                                                         pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    create_triangulation_and_dofs();
  }


  // @sect4{<code>NavierStokesProjection::create_triangulation_and_dofs</code>}

  template<int dim>
  void NavierStokesProjection<dim>::create_triangulation_and_dofs() {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::string   filename = "full_0.ucd";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);

    std::ifstream inlet_data("INLET_AIR_0.ucd");
    std::string curr_line;
    std::getline(inlet_data, curr_line);
    std::getline(inlet_data, curr_line);
    std::vector<double> xx(609), yy(609), zz(609);
    double index, x, y, z;
    for(unsigned int i = 0; i < 609; ++i) {
      std::getline(inlet_data, curr_line);
      std::istringstream iss(curr_line);
      iss.precision(16);
      iss >> index >> x >> y >> z;
      xx[i] = x;
      yy[i] = y;
      zz[i] = z;
    }
    for(const auto& cell: triangulation.active_cell_iterators()) {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        if(cell->face(f)->at_boundary()) {
          bool of_interest = true;
          for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
            unsigned int i = 0;
            while(i < 609) {
              if(std::abs(cell->face(f)->vertex(v)(0) - xx[i]) < 1e-9 &&
                 std::abs(cell->face(f)->vertex(v)(1) - yy[i]) < 1e-9 &&
                 std::abs(cell->face(f)->vertex(v)(2) - zz[i]) < 1e-9)
                 break;
              i++;
            }
            if(i == 609) {
              of_interest = false;
              break;
            }
          }
          if(of_interest)
            cell->face(f)->set_boundary_id(1);
        }
      }
    }

    std::ifstream outlet_data("OUTLET_AIR_0.ucd");
    std::getline(outlet_data, curr_line);
    std::getline(outlet_data, curr_line);
    xx.resize(3973);
    yy.resize(3973);
    zz.resize(3973);
    for(unsigned int i = 0; i < 3973; ++i) {
      std::getline(outlet_data, curr_line);
      std::istringstream iss(curr_line);
      iss.precision(16);
      iss >> index >> x >> y >> z;
      xx[i] = x;
      yy[i] = y;
      zz[i] = z;
    }
    for(const auto& cell: triangulation.active_cell_iterators()) {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        if(cell->face(f)->at_boundary()) {
          bool of_interest = true;
          for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v) {
            unsigned int i = 0;
            while(i < 3973) {
              if(std::abs(cell->face(f)->vertex(v)(0) - xx[i]) < 1e-8 &&
                 std::abs(cell->face(f)->vertex(v)(1) - yy[i]) < 1e-8 &&
                 std::abs(cell->face(f)->vertex(v)(2) - zz[i]) < 1e-8)
                 break;
              i++;
            }
            if(i == 3973) {
              of_interest = false;
              break;
            }
          }
          if(of_interest)
            cell->face(f)->set_boundary_id(2);
        }
      }
    }

    pcout<<"Grid read"<<std::endl;
    pcout<<"Number of global active cells = "<<triangulation.n_global_active_cells()<<std::endl;
    pcout<<"Boundary ids:"<<std::endl;
    const auto& boundary_ids = triangulation.get_boundary_ids();
    for(const auto& elem: boundary_ids)
      pcout<<elem<<std::endl;
    unsigned int number_of_boundary_cells = 0;
    for(const auto& cell: triangulation.active_cell_iterators()) {
      if(cell->at_boundary())
        number_of_boundary_cells++;
    }
    pcout<<"Number of boundary cells = "<<number_of_boundary_cells<<std::endl;

    GridOut grid_out;
    std::ofstream out_file("full_1.ucd");
    grid_out.write_ucd(triangulation, out_file);

    parallel::distributed::Triangulation<dim - 1, dim> triangulation_inlet(MPI_COMM_WORLD);
    GridGenerator::extract_boundary_mesh(triangulation, triangulation_inlet, {1});
    std::ofstream out_file_inlet("INLET_AIR_1.vtu");
    std::ofstream out_file_inlet_ucd("INLET_AIR_1.ucd");
    grid_out.write_vtu(triangulation_inlet, out_file_inlet);
    grid_out.write_ucd(triangulation_inlet, out_file_inlet_ucd);

    parallel::distributed::Triangulation<dim - 1, dim> triangulation_outlet(MPI_COMM_WORLD);
    GridGenerator::extract_boundary_mesh(triangulation, triangulation_outlet, {2});
    std::ofstream out_file_outlet("OUTLET_AIR_1.vtu");
    std::ofstream out_file_outlet_ucd("OUTLET_AIR_1.ucd");
    grid_out.write_vtu(triangulation_outlet, out_file_outlet);
    grid_out.write_ucd(triangulation_outlet, out_file_outlet_ucd);
  }

} // namespace Step35


// @sect3{ The main function }

// The main function looks very much like in all the other tutorial programs, so
// there is little to comment on here:
int main(int argc, char *argv[]) {
  try {
    using namespace Step35;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    NavierStokesProjection<3> test;

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

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
