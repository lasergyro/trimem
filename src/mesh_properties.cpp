/** \file mesh_properties.cpp
 * \brief Geometric properties and gradients on a OpenMesh::TriMesh.
 */
#include "mesh_properties.h"

#include "mesh_util.h"
#include "mesh_tether.h"
#include "mesh_repulsion.h"

namespace trimem {

VertexProperties vertex_properties(const TriMesh& mesh,
                                   const BondPotential& bonds,
                                   const SurfaceRepulsion& constraint,
                                   const VertexHandle& ve)
{
    VertexProperties p{ 0, 0, 0, 0, 0, 0 };

    for (auto he : mesh.voh_range(ve))
    {
        if ( not he.is_boundary() )
        {
            // edge curvature
            real el      = edge_length(mesh, he);
            real da      = dihedral_angle(mesh, he);
            p.curvature += 0.5 * el * da;

            // face area/volume
            p.area   += face_area(mesh, he);
            p.volume += face_volume(mesh, he);

            // bonds
            p.tethering += bonds.vertex_property(mesh, he);
        }
    }

    // correct multiplicity
    p.area      /= 3;
    p.volume    /= 3;
    p.curvature /= 2;
    p.tethering /= bonds.valence();

    p.bending    = 2 * p.curvature * p.curvature / p.area;

    // mesh repulsion
    p.repulsion = constraint.vertex_property(mesh, ve.idx());

    return p;
}

void vertex_properties_grad(const TriMesh& mesh,
                            const BondPotential& bonds,
                            const SurfaceRepulsion& repulse,
                            const VertexHandle& ve,
                            std::vector<VertexPropertiesGradient>& d_props)
{
    // pre-compute vertex-curvature to vertex-area ratio (needed for bending)
    real curvature = 0;
    real area      = 0;
    for (auto he : mesh.voh_range(ve))
    {
        real l  = edge_length(mesh, he);
        real da = dihedral_angle(mesh, he);

        curvature += 0.5 * l * da;
        area      += face_area(mesh, he);
    }
    real c_to_a = curvature / area * 1.5;

    for (auto he : mesh.voh_range(ve))
    {
        if ( not he.is_boundary() )
        {
            //relevant vertex indices of the facet-pair associated to 'he'
            std::vector<int> idx
              { ve.idx(),
                mesh.to_vertex_handle(he).idx(),
                mesh.to_vertex_handle(mesh.next_halfedge_handle(he)).idx(),
                mesh.to_vertex_handle(mesh.next_halfedge_handle(
                  mesh.opposite_halfedge_handle(he))).idx()
              };

            // edge curvature
            real edge_length = trimem::edge_length(mesh, he);
            real edge_angle  = trimem::dihedral_angle(mesh, he);
            auto d_length = trimem::edge_length_grad(mesh, he);
            auto d_angle  = trimem::dihedral_angle_grad(mesh, he);
            for (size_t i=0; i<d_length.size(); i++)
            {
                auto val = 0.25 * edge_angle * d_length[i];
                for (int j=0; j<3; j++)
                {
#pragma omp atomic
                    d_props[idx[i]].curvature[j] += val[j];

                    // contribution to bending
#pragma omp atomic
                    d_props[idx[i]].bending[j] += 4.0 * val[j] * c_to_a;
                }
            }

            for (size_t i=0; i<d_angle.size(); i++)
            {
                auto val = 0.25 * edge_length * d_angle[i];
                for (int j=0; j<3; j++)
                {
#pragma omp atomic
                    d_props[idx[i]].curvature[j] += val[j];

                    // contribution to bending
#pragma omp atomic
                    d_props[idx[i]].bending[j] += 4.0 * val[j] * c_to_a;
                }
            }

            // face area
            auto d_face_area = trimem::face_area_grad(mesh, he);
            for (size_t i=0; i<d_face_area.size(); i++)
            {
                auto val = d_face_area[i] / 3;
                for (int j=0; j<3; j++)
                {
#pragma omp atomic
                    d_props[idx[i]].area[j] += val[j];

                    // contribution to bending
#pragma omp atomic
                    d_props[idx[i]].bending[j] -= 2.0 * val[j] * c_to_a * c_to_a;
                }
            }

            // face volume
            auto d_face_volume = trimem::face_volume_grad(mesh, he);
            for (size_t i=0; i<d_face_volume.size(); i++)
            {
                auto val = d_face_volume[i] / 3;
                for (int j=0; j<3; j++)
                {
#pragma omp atomic
                    d_props[idx[i]].volume[j] += val[j];
                }
            }

            // tether bonds
            auto d_bond = bonds.vertex_property_grad(mesh, he);
            for (size_t i=0; i<d_bond.size(); i++)
            {
                auto val = d_bond[i] / bonds.valence();
                for (int j=0; j<3; j++)
                {
#pragma omp atomic
                    d_props[idx[i]].tethering[j] += val[j];
                }
            }
        } // not boundary
    } // outgoing halfedges

    // repulsion penalty
    std::vector<Point> d_repulse;
    std::vector<int> idx;
    std::tie(d_repulse, idx) = repulse.vertex_property_grad(mesh, ve.idx());
    for (size_t i=0; i<d_repulse.size(); i++)
    {
        auto val = d_repulse[i];
        for (int j=0; j<3; j++)
        {
#pragma omp atomic
            d_props[idx[i]].repulsion[j] -= val[j];
#pragma omp atomic
            d_props[ve.idx()].repulsion[j] += val[j];
        }
    }

}

void expose_properties(py::module& m)
{
    py::class_<VertexProperties>(
        m,
        "VertexProperties",
        R"pbdoc(Container for vertex-averaged properties.

        The vertex averaged quantities `area`, `volume`, `curvature` and
        `bending energy` are at the core of the evaluations involved in the
        Helfrich functional that is represented by :class:`EnergyManager`.
        class:`VertexProperties` encapsulates access to these quantities. It
        further allows for the automatic reduction on vectors of
        :class:`VertexProperties` by implementing the operators
        ``+=`` and ``-=``.

        Its current usage in trimem implies an AoS approach that might be
        replaced by an SoA approach if required by performance considerations.
        In this case the idea is to still keep the container 'interface' as
        is and manage the data layout behind the scenes.
        )pbdoc"
        )
        .def(py::init())
        .def_readwrite(
            "area",
            &VertexProperties::area,
            "Face area"
        )
        .def_readwrite(
            "volume",
            &VertexProperties::volume,
            "Face volume")
        .def_readwrite(
            "curvature",
            &VertexProperties::curvature,
            "Edge curvature")
        .def_readwrite(
            "bending",
            &VertexProperties::bending,
            "Bending energy")
        .def_readwrite(
            "tethering",
            &VertexProperties::tethering,
            "Tether regularization")
        .def_readwrite(
            "repulsion",
            &VertexProperties::repulsion,
            "Repulsion penalty")
        //TODO: remove pickling support since unused
        .def(py::pickle(
           [](const VertexProperties &p) { // __getstate__
               return py::make_tuple(
                   p.area,
                   p.volume,
                   p.curvature,
                   p.bending,
                   p.tethering,
                   p.repulsion);
           },
           [](py::tuple t) { // __setstate__
               if (t.size() != 6)
                   throw std::runtime_error("Invalid state!");
               VertexProperties p = {
                   t[0].cast<real>(),
                   t[1].cast<real>(),
                   t[2].cast<real>(),
                   t[3].cast<real>(),
                   t[4].cast<real>(),
                   t[5].cast<real>() };
               return p;
           }));
}
}
