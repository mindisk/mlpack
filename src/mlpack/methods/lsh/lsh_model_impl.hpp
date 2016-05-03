
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_IMPL_HPP

#include <mlpack/core.hpp>

#include "hash_model.hpp"

//#include "lsh_model.hpp"

namespace mlpack
{
    namespace neighbor
    {
        template<typename SortPolicy>
        lshModel<SortPolicy>::lshModel(const arma::mat& referenceSet,
                                        const size_t hashType,
                                        const size_t secondHashSize,
                                        const size_t bucketSize,
                
                                        const size_t numProj,
                                        const size_t numTables,
                                        const double hashWidthIn,
                                        
                                        const size_t dimensions,
                                        const size_t planes) :
        referenceSet(NULL), // This will be set in Train().
        hashType(hashType),
        secondHashSize(secondHashSize),
        bucketSize(bucketSize),
                
        numProj(numProj),
        numTables(numTables),
        hashWidth(hashWidthIn),
        
        numDimensions(dimensions),
        numPlanes(planes),
        
        ownsSet(false),
        distanceEvaluations(0)
        {
            // Pass work to training function.
            Train(referenceSet, hashType, secondHashSize, bucketSize, numProj, numTables, hashWidthIn, dimensions, planes);                
        }
        // Empty constructor.
        template<typename SortPolicy>
        lshModel<SortPolicy>::lshModel() :
        referenceSet(new arma::mat()), // empty dataset
        hashType(1),
        secondHashSize(99901),
        bucketSize(500),
                       
        numProj(0),
        numTables(0),
        hashWidth(0),
        
        numDimensions(1),
        numPlanes(1),
                
        ownsSet(true),
        distanceEvaluations(0)
        {
        }

        // Destructor.
        template<typename SortPolicy>
        lshModel<SortPolicy>::~lshModel() 
        {
            if (ownsSet)
            {
                delete referenceSet;
            }
        }
        
        template<typename SortPolicy>
        void lshModel<SortPolicy>::Train(const arma::mat& referenceSet,
                                         const size_t hashType,
                                         const size_t secondHashSize,
                                         const size_t bucketSize,
                
                                         const size_t numProj,
                                         const size_t numTables,
                                         const double hashWidthIn,
                
                                         const size_t dimensions,
                                         const size_t planes) 
        {
            // Set new reference set.
            if (this->referenceSet && ownsSet)
            {
                delete this->referenceSet;
            }
            this->ownsSet = false;
            this->referenceSet = &referenceSet;
            this->hashType = hashType;
            this->secondHashSize = secondHashSize;
            this->bucketSize = bucketSize;          

            // Set new parameters.
            this->numProj = numProj;
            this->numTables = numTables;
            this->hashWidth = hashWidthIn;
            
            this->numDimensions = dimensions;
            this->numPlanes = planes;           
            
            hash.setParams(referenceSet, hashType, secondHashSize, bucketSize, numProj, numTables, hashWidthIn, dimensions, planes);
//            hash(referenceSet, hashType, secondHashSize, bucketSize, numProj, numTables, hashWidthIn, dimensions, planes);

            if (hashWidth == 0.0) // The user has not provided any value.
            {
                // Compute a heuristic hash width from the data.
                for (size_t i = 0; i < 25; i++) 
                {
                    size_t p1 = (size_t) math::RandInt(referenceSet.n_cols);
                    size_t p2 = (size_t) math::RandInt(referenceSet.n_cols);

                    hashWidth += std::sqrt(metric::EuclideanDistance::Evaluate(referenceSet.unsafe_col(p1), referenceSet.unsafe_col(p2)));
                }
                hashWidth /= 25;
            }

            Log::Info << "Hash width chosen as: " << hashWidth << std::endl;

            hash.BuildHash();
        }

        // Search for nearest neighbors in a given query set.
        template<typename SortPolicy>
        void lshModel<SortPolicy>::Search(const arma::mat& querySet, const size_t k, arma::Mat<size_t>& resultingNeighbors, arma::mat& distances, const size_t numTablesToSearch) 
        {
            // Ensure the dimensionality of the query set is correct.
            if (querySet.n_rows != referenceSet->n_rows) 
            {
                std::ostringstream oss;
                oss << "lshModel::Search(): dimensionality of query set ("<< querySet.n_rows << ") is not equal to the dimensionality the model was trained on (" << referenceSet->n_rows << ")!" << std::endl;
                throw std::invalid_argument(oss.str());
            }

            if (k > referenceSet->n_cols) 
            {
                std::ostringstream oss;
                oss << "lshModel::Search(): requested " << k << " approximate nearest neighbors, but reference set has " << referenceSet->n_cols<< " points!" << std::endl;
                throw std::invalid_argument(oss.str());
            }

            // Set the size of the neighbor and distance matrices.
            resultingNeighbors.set_size(k, querySet.n_cols);
            distances.set_size(k, querySet.n_cols);
            distances.fill(SortPolicy::WorstDistance());
            resultingNeighbors.fill(referenceSet->n_cols);

            // If the user asked for 0 nearest neighbors... uh... we're done.
            if (k == 0)
            {
                return;
            }
            size_t avgIndicesReturned = 0;

            Timer::Start("computing_neighbors");

            // Go through every query point sequentially.
            for (size_t i = 0; i < querySet.n_cols; i++) 
            {
                // Hash every query into every hash table and eventually into the 'secondHashTable' to obtain the neighbor candidates.
                arma::uvec refIndices;
                ReturnIndicesFromTable(querySet.col(i), refIndices, numTablesToSearch);

                // An informative book-keeping for the number of neighbor candidates returned on average.
                avgIndicesReturned += refIndices.n_elem;

                // Sequentially go through all the candidates and save the best 'k' candidates.
                for (size_t j = 0; j < refIndices.n_elem; j++)
                {
                    BaseCase(i, (size_t) refIndices[j], querySet, resultingNeighbors, distances);
                }
            }

            Timer::Stop("computing_neighbors");

            distanceEvaluations += avgIndicesReturned;
            avgIndicesReturned /= querySet.n_cols;
            Log::Info << avgIndicesReturned << " distinct indices returned on average." << std::endl;
        }

        // Search for approximate neighbors of the reference set.
        template<typename SortPolicy>
        void lshModel<SortPolicy>::Search(const size_t k, arma::Mat<size_t>& resultingNeighbors, arma::mat& distances, const size_t numTablesToSearch) 
        {
            // This is monochromatic search; the query set is the reference set.
            resultingNeighbors.set_size(k, referenceSet->n_cols);
            distances.set_size(k, referenceSet->n_cols);
            distances.fill(SortPolicy::WorstDistance());
            resultingNeighbors.fill(referenceSet->n_cols);

            size_t avgIndicesReturned = 0;

            Timer::Start("computing_neighbors");

            // Go through every query point sequentially.
            for (size_t i = 0; i < referenceSet->n_cols; i++) 
            {
                // Hash every query into every hash table and eventually into the 'secondHashTable' to obtain the neighbor candidates.
                arma::uvec refIndices;
                ReturnIndicesFromTable(referenceSet->col(i), refIndices, numTablesToSearch);

                // An informative book-keeping for the number of neighbor candidates returned on average.
                avgIndicesReturned += refIndices.n_elem;

                // Sequentially go through all the candidates and save the best 'k' candidates.
                for (size_t j = 0; j < refIndices.n_elem; j++)
                {
                    BaseCase(i, (size_t) refIndices[j], resultingNeighbors, distances);
                }
            }

            Timer::Stop("computing_neighbors");

            distanceEvaluations += avgIndicesReturned;
            avgIndicesReturned /= referenceSet->n_cols;
            Log::Info << avgIndicesReturned << " distinct indices returned on average." << std::endl;
        }
        template<typename SortPolicy>
        void lshModel<SortPolicy>::InsertNeighbor(arma::mat& distances, arma::Mat<size_t>& neighbors, const size_t queryIndex, const size_t pos, const size_t neighbor, const double distance) const 
        {
            // We only MEMMOVE() if there is actually a need to shift something.
            if (pos < (distances.n_rows - 1)) 
            {
                const size_t len = (distances.n_rows - 1) - pos;
                memmove(distances.colptr(queryIndex) + (pos + 1), distances.colptr(queryIndex) + pos, sizeof (double) * len);
                memmove(neighbors.colptr(queryIndex) + (pos + 1), neighbors.colptr(queryIndex) + pos, sizeof (size_t) * len);
            }

            // Now put the new information in the right index.
            distances(pos, queryIndex) = distance;
            neighbors(pos, queryIndex) = neighbor;
        }

        // Base case where the query set is the reference set.  (So, we can't return ourselves as the nearest neighbor.)
        template<typename SortPolicy>
        void lshModel<SortPolicy>::BaseCase(const size_t queryIndex, const size_t referenceIndex, arma::Mat<size_t>& neighbors, arma::mat& distances) const 
        {
            // If the points are the same, we can't continue.
            if (queryIndex == referenceIndex)
            {
                return;
            }

            const double distance = metric::EuclideanDistance::Evaluate(referenceSet->unsafe_col(queryIndex), referenceSet->unsafe_col(referenceIndex));

            // If this distance is better than any of the current candidates, the
            // SortDistance() function will give us the position to insert it into.
            arma::vec queryDist = distances.unsafe_col(queryIndex);
            arma::Col<size_t> queryIndices = neighbors.unsafe_col(queryIndex);
            size_t insertPosition = SortPolicy::SortDistance(queryDist, queryIndices, distance);

            // SortDistance() returns (size_t() - 1) if we shouldn't add it.
            if (insertPosition != (size_t() - 1))
            {
                InsertNeighbor(distances, neighbors, queryIndex, insertPosition, referenceIndex, distance);
            }
        }

        // Base case for bichromatic search.
        template<typename SortPolicy>
        void lshModel<SortPolicy>::BaseCase(const size_t queryIndex, const size_t referenceIndex, const arma::mat& querySet, arma::Mat<size_t>& neighbors, arma::mat& distances) const 
        {
            const double distance = metric::EuclideanDistance::Evaluate(querySet.unsafe_col(queryIndex), referenceSet->unsafe_col(referenceIndex));

            // If this distance is better than any of the current candidates, the
            // SortDistance() function will give us the position to insert it into.
            arma::vec queryDist = distances.unsafe_col(queryIndex);
            arma::Col<size_t> queryIndices = neighbors.unsafe_col(queryIndex);
            size_t insertPosition = SortPolicy::SortDistance(queryDist, queryIndices, distance);

            // SortDistance() returns (size_t() - 1) if we shouldn't add it.
            if (insertPosition != (size_t() - 1))
            {
                InsertNeighbor(distances, neighbors, queryIndex, insertPosition, referenceIndex, distance);
            }
        }

        template<typename SortPolicy>
        template<typename VecType>       
        void lshModel<SortPolicy>::ReturnIndicesFromTable(const VecType& queryPoint, arma::uvec& referenceIndices, size_t numTablesToSearch) const 
        {
            // Decide on the number of tables to look into.
            if (numTablesToSearch == 0) // If no user input is given, search all.
            {
                numTablesToSearch = numTables;
            }

            // Sanity check to make sure that the existing number of tables is not
            // exceeded.
            if (numTablesToSearch > numTables)
            {
                numTablesToSearch = numTables;
            }
//            // Hash the query in each of the 'numTablesToSearch' hash tables using the
//            // 'numProj' projections for each table. This gives us 'numTablesToSearch'
//            // keys for the query where each key is a 'numProj' dimensional integer
//            // vector.
//            // Compute the projection of the query in each table.
//            arma::mat allProjInTables(numProj, numTablesToSearch);
//            for (size_t i = 0; i < numTablesToSearch; i++)
//            {
//                allProjInTables.unsafe_col(i) = projections[i].t() * queryPoint;
//            }
//            allProjInTables += offsets.cols(0, numTablesToSearch - 1);
//            allProjInTables /= hashWidth;
//
//            // Compute the hash value of each key of the query into a bucket of the
//            // 'secondHashTable' using the 'secondHashWeights'.
//            arma::rowvec hashVec = secondHashWeights.t() * arma::floor(allProjInTables);
//
//            for (size_t i = 0; i < hashVec.n_elem; i++)
//            {
//                hashVec[i] = (double) ((size_t) hashVec[i] % secondHashSize);
//            }
//            Log::Assert(hashVec.n_elem == numTablesToSearch);

            
            arma::rowvec hashVec = hash.hashQuery(queryPoint, numTablesToSearch);
            // For all the buckets that the query is hashed into, sequentially
            // collect the indices in those buckets.
            arma::Col<size_t> refPointsConsidered;
            refPointsConsidered.zeros(referenceSet->n_cols);

            for (size_t i = 0; i < hashVec.n_elem; i++) // For all tables.
            {
                size_t hashInd = (size_t) hashVec[i];

                if (bucketContentSize[hashInd] > 0) 
                {
                    // Pick the indices in the bucket corresponding to 'hashInd'.
                    size_t tableRow = bucketRowInHashTable[hashInd];
                    assert(tableRow < secondHashSize);
                    assert(tableRow < secondHashTable.n_rows);

                    for (size_t j = 0; j < bucketContentSize[hashInd]; j++)
                    {
                        refPointsConsidered[secondHashTable(tableRow, j)]++;
                    }
                }
            }

            referenceIndices = arma::find(refPointsConsidered > 0);
        }

        template<typename SortPolicy>
        template<typename Archive>       
        void lshModel<SortPolicy>::Serialize(Archive& ar, const unsigned int /* version */) 
        {
            using data::CreateNVP;

            // If we are loading, we are going to own the reference set.
            if (Archive::is_loading::value) 
            {
                if (ownsSet)
                {
                    delete referenceSet;
                }
                ownsSet = true;
            }
            ar & CreateNVP(referenceSet, "referenceSet");
            ar & CreateNVP(hashType, "hashType");
            ar & CreateNVP(numProj, "numProj");
            ar & CreateNVP(numTables, "numTables");

            // Delete existing projections, if necessary.
            if (Archive::is_loading::value)
            {
                projections.clear();
            }
            ar & CreateNVP(projections, "projections");
            ar & CreateNVP(offsets, "offsets");
            ar & CreateNVP(hashWidth, "hashWidth");
            ar & CreateNVP(secondHashSize, "secondHashSize");
            ar & CreateNVP(secondHashWeights, "secondHashWeights");
            ar & CreateNVP(bucketSize, "bucketSize");
            ar & CreateNVP(secondHashTable, "secondHashTable");
            ar & CreateNVP(bucketContentSize, "bucketContentSize");
            ar & CreateNVP(bucketRowInHashTable, "bucketRowInHashTable");
            ar & CreateNVP(distanceEvaluations, "distanceEvaluations");
            ar & CreateNVP(numPlanes, "numPlanes");
            ar & CreateNVP(numDimensions, "numDimensions");
            ar & CreateNVP(hash, "hash");
        }
    }
}


#endif /* LSH_MODEL_IMPL_HPP */

