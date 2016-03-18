
#ifndef __MLPACK_METHODS_NEIGHBOR_HASH_FUNCTION_H
#define __MLPACK_METHODS_NEIGHBOR_HASH_FUNCTION_H

#include <mlpack/core.hpp>
#include <vector>
#include <string>

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>


namespace mlpack {
    namespace neighbor {

        template<typename SortPolicy = NearestNeighborSort>
        class Hash_Function {
        public:
            /**
             * This function builds a hash table with two levels of hashing as presented
             * in the paper. This function first hashes the points with 'numProj' random
             * projections to a single hash table creating (key, point ID) pairs where the
             * key is a 'numProj'-dimensional integer vector.
             *
             * Then each key in this hash table is hashed into a second hash table using a
             * standard hash.
             *
             * This function does not have any parameters and relies on parameters which
             * are private members of this class, initialized during the class
             * initialization.
             */
            /**
             * A signature for building hash.
             * The output is a matrix of type size_t
             */
            void BuildHash(const arma::mat* referenceSet,
                    const size_t numProj,
                    const size_t numTables,
                    const double hashWidthIn,
                    const size_t secondHashSize,
                    const size_t bucketSize);

            arma::Mat<size_t> getSecondHashTable() {
                return secondHashTable;
            }

            arma::vec getSecondHashTableWeights() {
                return secondHashWeights;
            }

            arma::Col<size_t> getBucketContentSize() {
                return bucketContentSize;
            }

            arma::Col<size_t> getBucketRowInHashTable() {
                return bucketRowInHashTable;
            }


            //! The weights of the second hash.
            arma::vec secondHashWeights;

            //! The final hash table; should be (< secondHashSize) x bucketSize.
            arma::Mat<size_t> secondHashTable;

            //! The number of elements present in each hash bucket; should be
            //! secondHashSize.
            arma::Col<size_t> bucketContentSize;

            //! For a particular hash value, points to the row in secondHashTable
            //! corresponding to this value.  Should be secondHashSize.
            arma::Col<size_t> bucketRowInHashTable;

            //! The std::vector containing the projection matrix of each table.
            std::vector<arma::mat> projections;

            //! The list of the offsets 'b' for each of the projection for each table.
            arma::mat offsets;

        };
    }
}
#include "hash_function_impl.hpp"
#endif // HASH_FUNCTION_H

