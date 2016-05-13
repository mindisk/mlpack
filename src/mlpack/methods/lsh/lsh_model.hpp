#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_HPP

#include <mlpack/core.hpp>
#include <vector>
#include <string>

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>

#include "hash_model.hpp"

namespace mlpack 
{
    namespace neighbor 
    {

        /**
         * The lshModel class; this class builds a hash on the reference set and uses
         * this hash to compute the distance-approximate nearest-neighbors of the given
         * queries.
         *
         * @tparam SortPolicy The sort policy for distances; see NearestNeighborSort.
         */
        template<typename SortPolicy = NearestNeighborSort>
        class lshModel 
        {
        public:
            /**
             * This function initializes the LSH class. It builds the hash on the
             * reference set with 2-stable distributions. See the individual functions
             * performing the hashing for details on how the hashing is done.
             *
             * @param referenceSet Set of reference points and the set of queries.
             * @param numProj Number of projections in each hash table (anything between
             *     10-50 might be a decent choice).
             * @param numTables Total number of hash tables (anything between 10-20
             *     should suffice).
             * @param hashWidth The width of hash for every table. If 0 (the default) is
             *     provided, then the hash width is automatically obtained by computing
             *     the average pairwise distance of 25 pairs.  This should be a reasonable
             *     upper bound on the nearest-neighbor distance in general.
             * @param secondHashSize The size of the second hash table. This should be a
             *     large prime number.
             * @param bucketSize The size of the bucket in the second hash table. This is
             *     the maximum number of points that can be hashed into single bucket.
             *     Default values are already provided here.
             */
            lshModel(const arma::mat& referenceSet,
                     const size_t hashType = 1,
                     const size_t secondHashSize = 99901,
                     const size_t bucketSize = 500,
                    
                     const size_t numProj = 10,
                     const size_t numTables = 30,
                     const double hashWidth = 0.0,
                    
                     const size_t dimensions = 1,
                     const size_t planes = 1,
                     const size_t shears = 1);
            /**
             * Create an untrained LSH model.  Be sure to call Train() before calling
             * Search(); otherwise, an exception will be thrown when Search() is called.
             */
            lshModel();

            /**
             * Clean memory.
             */
            ~lshModel();

            /**
             * Train the LSH model on the given dataset.  This means building new hash
             * tables.
             */
            void Train(const arma::mat& referenceSet,
                       const size_t hashType,
                       const size_t secondHashSize = 99901,
                       const size_t bucketSize = 500,
            
                       const size_t numProj = 10,
                       const size_t numTables = 30,
                       const double hashWidth = 0.0,
                    
                       const size_t dimensions = 1,
                       const size_t planes = 1,
                       const size_t shears = 1);

            /**
             * Compute the nearest neighbors of the points in the given query set and
             * store the output in the given matrices.  The matrices will be set to the
             * size of n columns by k rows, where n is the number of points in the query
             * dataset and k is the number of neighbors being searched for.
             *
             * @param querySet Set of query points.
             * @param k Number of neighbors to search for.
             * @param resultingNeighbors Matrix storing lists of neighbors for each query
             *     point.
             * @param distances Matrix storing distances of neighbors for each query
             *     point.
             * @param numTablesToSearch This parameter allows the user to have control
             *     over the number of hash tables to be searched. This allows
             *     the user to pick the number of tables it can afford for the time
             *     available without having to build hashing for every table size.
             *     By default, this is set to zero in which case all tables are
             *     considered.
             */
            void Search(const arma::mat& querySet,
                        const size_t k,
                        arma::Mat<size_t>& resultingNeighbors,
                        arma::mat& distances,
                        const size_t numTablesToSearch = 0);

            /**
             * Compute the nearest neighbors and store the output in the given matrices.
             * The matrices will be set to the size of n columns by k rows, where n is
             * the number of points in the query dataset and k is the number of neighbors
             * being searched for.
             *
             * @param k Number of neighbors to search for.
             * @param resultingNeighbors Matrix storing lists of neighbors for each query
             *     point.
             * @param distances Matrix storing distances of neighbors for each query
             *     point.
             * @param numTablesToSearch This parameter allows the user to have control
             *     over the number of hash tables to be searched. This allows
             *     the user to pick the number of tables it can afford for the time
             *     available without having to build hashing for every table size.
             *     By default, this is set to zero in which case all tables are
             *     considered.
             */
            void Search(const size_t k,
                        arma::Mat<size_t>& resultingNeighbors,
                        arma::mat& distances,
                        const size_t numTablesToSearch = 0);

            /**
             * Serialize the LSH model.
             *
             * @param ar Archive to serialize to.
             */
            template<typename Archive>
            void Serialize(Archive& ar, const unsigned int /* version */);

            //! Return the reference dataset.
            const arma::mat& ReferenceSet() const 
            {
                return *referenceSet;
            }
            
            //! Modify the hash type used.
            size_t& HashType()
            {
                return hashType;
            }
            
            //! Get the type of hash used
            size_t HashType() const 
            {
                return hashType;
            }
            
            //! Modify the number of dimensions.
            size_t& Dimensions()
            {
                return numDimensions;
            }

            size_t Dimensions() const
            {
                return numDimensions;
            }
            //! Modify the number of planes.
            size_t& Planes()
            {
                return numPlanes;
            }
            size_t Planes() const
            {
                return numPlanes;
            }
            //! Return the number of distance evaluations performed.
            size_t DistanceEvaluations() const 
            {
                return distanceEvaluations;
            }
            
            //! Modify the number of distance evaluations performed.
            size_t& DistanceEvaluations() 
            {
                return distanceEvaluations;
            }        

            //! Get the bucket size of the second hash.
            size_t BucketSize() const 
            {
                return bucketSize;
            }

           
        private:
            /**
             * This function takes a query and hashes it into each of the hash tables to
             * get keys for the query and then the key is hashed to a bucket of the second
             * hash table and all the points (if any) in those buckets are collected as
             * the potential neighbor candidates.
             *
             * @param queryPoint The query point currently being processed.
             * @param referenceIndices The list of neighbor candidates obtained from
             *    hashing the query into all the hash tables and eventually into
             *    multiple buckets of the second hash table.
             */
            template<typename VecType>
            void ReturnIndicesFromTable(const VecType& queryPoint, arma::uvec& referenceIndices, size_t numTablesToSearch) const;

            /**
             * This is a helper function that computes the distance of the query to the
             * neighbor candidates and appropriately stores the best 'k' candidates.  This
             * is specific to the monochromatic search case, where the query set is the
             * reference set.
             *
             * @param queryIndex The index of the query in question
             * @param referenceIndex The index of the neighbor candidate in question
             * @param neighbors Matrix holding output neighbors.
             * @param distances Matrix holding output distances.
             */
            void BaseCase(const size_t queryIndex, const size_t referenceIndex, arma::Mat<size_t>& neighbors, arma::mat& distances) const;

            /**
             * This is a helper function that computes the distance of the query to the
             * neighbor candidates and appropriately stores the best 'k' candidates.  This
             * is specific to bichromatic search, where the query set is not the same as
             * the reference set.
             *
             * @param queryIndex The index of the query in question
             * @param referenceIndex The index of the neighbor candidate in question
             * @param querySet Set of query points.
             * @param neighbors Matrix holding output neighbors.
             * @param distances Matrix holding output distances.
             */
            void BaseCase(const size_t queryIndex, const size_t referenceIndex, const arma::mat& querySet, arma::Mat<size_t>& neighbors, arma::mat& distances) const;

            /**
             * This is a helper function that efficiently inserts better neighbor
             * candidates into an existing set of neighbor candidates. This function is
             * only called by the 'BaseCase' function.
             *
             * @param distances Matrix holding output distances.
             * @param neighbors Matrix holding output neighbors.
             * @param queryIndex This is the index of the query being processed currently
             * @param pos The position of the neighbor candidate in the current list of
             *    neighbor candidates.
             * @param neighbor The neighbor candidate that is being inserted into the list
             *    of the best 'k' candidates for the query in question.
             * @param distance The distance of the query to the neighbor candidate.
             */
            void InsertNeighbor(arma::mat& distances,
                    arma::Mat<size_t>& neighbors,
                    const size_t queryIndex,
                    const size_t pos,
                    const size_t neighbor,
                    const double distance) const;

            const arma::mat* referenceSet;          //! Reference dataset.
            hashModel hash;                         //! instance that will be used to build the hash tables                              
            size_t hashType;                        //! the hash type to be used to create the hash model
            size_t secondHashSize;                  //! The big prime representing the size of the second hash.
            size_t bucketSize;                      //! The bucket size of the second hash.        
            
            size_t numProj;                         //! The number of projections.           
            size_t numTables;                       //! The number of hash tables.
            double hashWidth;                       //! The hash width.
            
            size_t numDimensions;                   //! dimensionality
            size_t numPlanes;                       //! number of planes 
            
            bool ownsSet;                           //! If true, we own the reference set.
            size_t distanceEvaluations;             //! The number of distance evaluations.   
            
            size_t shears;
        }; // class lshModel

    } // namespace neighbor
} // namespace mlpack

#include "lsh_model_impl.hpp"

#endif /* LSH_MODEL_HPP */

