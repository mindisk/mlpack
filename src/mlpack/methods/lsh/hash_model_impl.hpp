#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_HASH_MODEL_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_HASH_MODEL_IMPL_HPP

#include <mlpack/core.hpp>
#include <cmath>
#include "hash_model.hpp"
#include <iostream>
using namespace std;


namespace mlpack 
{
    namespace neighbor 
    {
        const size_t hashModel::minHashType = 1;
        const size_t hashModel::maxHashType = 3;
                  
        hashModel::hashModel() 
        {
        }

        hashModel::~hashModel() 
        {
        }
        void hashModel::setParams(const arma::mat& referenceSet,
                            const size_t hashType,
                            const size_t secondHashSize,
                            const size_t bucketSize,
                
                            const size_t numProj,
                            const size_t numTables,
                            const double hashWidthIn,

                            const size_t dimensions,
                            const size_t planes,
                            const size_t shears)
        {
            this->referenceSet = &referenceSet;
            this->hashType = hashType;
            this->secondHashSize = secondHashSize;
            this->bucketSize = bucketSize;
            
            this->numProj = numProj;
            this->numTables = numTables;
            this->hashWidth = hashWidthIn;
            
            this->numDimensions = dimensions;
            this->numPlanes = planes;
            
            this->numShears = shears;
        }
            // Build hash method: create the first level hash and the second level hash

        void hashModel::BuildHash() 
        {
            // The first level hash for a single table outputs a 'numProj'-dimensional integer key for each point in the set -- (key, pointID)
            //
            // The key creation details are presented below
            //
            // The second level hash is performed by hashing the key to an integer in the range [0, 'secondHashSize').
            // This is done by creating a weight vector 'secondHashWeights' of length 'numProj' with each entry an integer randomly chosen between [0, 'secondHashSize').
            //
            // Then the bucket for any key and its corresponding point is given by <key, 'secondHashWeights'> % 'secondHashSize' and the corresponding point ID is put into that bucket.
            
            // Step I: Prepare the second level hash.
           

            // The 'secondHashTable' is initially an empty matrix of size ('secondHashSize' x 'bucketSize'). 
            // But by only filling the buckets as points land in them allows us to shrink the size of the 'secondHashTable' at the end of the hashing.

            // Fill the second hash table n = referenceSet.n_cols.  
            // This is because no point has index 'n' so the presence of this in the bucket denotes that there are no more points in this bucket.
            secondHashTable.set_size(secondHashSize, bucketSize);
            secondHashTable.fill(referenceSet->n_cols);

            // Keep track of the size of each bucket in the hash.  At the end of hashing most buckets will be empty.
            bucketContentSize.zeros(secondHashSize);

            // Instead of putting the points in the row corresponding to the bucket, we chose the next empty row and keep track of the row in which the bucket lies.
            // This allows us to stack together and slice out the empty buckets at the end of the hashing.
            bucketRowInHashTable.set_size(secondHashSize);
            bucketRowInHashTable.fill(secondHashSize);

            // Keep track of number of non-empty rows in the 'secondHashTable'.
            size_t numRowsInTable = 0;
            
            
            switch (hashType) 
            {
                case 1:
                    // Obtain the weights for the second hash.
                    secondHashWeights = arma::floor(arma::randu(numProj) * (double) secondHashSize);
            
                    // Step II: The offsets for all projections in all tables.
                    // Since the 'offsets' are in [0, hashWidth], we obtain the 'offsets' as randu(numProj, numTables) * hashWidth.
                    offsets.randu(numProj, numTables);
                    offsets *= hashWidth;
                    // Step III: Create each hash table in the first level hash one by one and putting them directly into the 'secondHashTable' for memory efficiency.
                    projections.clear(); // Reset projections vector.
                    hashType2StableDistribution(&numRowsInTable);
                    break;
                case 2:
                    // Obtain the weights for the second hash.
                    secondHashWeights = arma::floor(arma::randu(numPlanes) * (double) secondHashSize);
                    this->planes.clear();
                    for (size_t i = 0; i < numTables; i++)
                    {
                      arma::mat planeMat;
                      planeMat.randn(numDimensions, numPlanes);

                      this->planes.push_back(planeMat);
                    }
                    for(size_t i = 0; i < referenceSet->n_cols; i++)
                    {
                        arma::vec query = referenceSet->col(i);
                        arma::mat hyperplaneMat = hashTypeHyperplaneOnePoint(query, numTables);
                        //hyperplaneMat.raw_print(std::cout, "H =");
                        // Step VI: Putting the points in the 'secondHashTable' by hashing the key.
                        // Now we hash every key, point ID to its corresponding bucket.
                        arma::rowvec secondHashVec = secondHashWeights.t() * arma::floor(hyperplaneMat);

                        // This gives us the bucket for the corresponding point ID.
                        for (size_t j = 0; j < secondHashVec.n_elem; j++)
                        {
                            secondHashVec[j] = (double) ((size_t) secondHashVec[j] % secondHashSize);
                        }
                        //Log::Assert(secondHashVec.n_elem == referenceSet->n_cols);

                        // Insert the point in the corresponding row to its bucket in the 'secondHashTable'.
                        for (size_t j = 0; j < secondHashVec.n_elem; j++) 
                        {
                            // This is the bucket number.
                            size_t hashInd = (size_t) secondHashVec[j];
                            // The point ID is 'j'.

                            // If this is currently an empty bucket, start a new row keep track of which row corresponds to the bucket.
                            if (bucketContentSize[hashInd] == 0) 
                            {
                                // Start a new row for hash.
                                bucketRowInHashTable[hashInd] = numRowsInTable;
                                secondHashTable(numRowsInTable, 0) = i;
                                numRowsInTable++;
                            }
                            else 
                            {
                                // If bucket is already present in the 'secondHashTable', find the corresponding row and insert the point ID in this row unless the bucket is full, in which case, do nothing.
                                if (bucketContentSize[hashInd] < bucketSize)
                                {
                                    secondHashTable(bucketRowInHashTable[hashInd], bucketContentSize[hashInd]) = i;
                                }
                            }

                            // Increment the count of the points in this bucket.
                            if (bucketContentSize[hashInd] < bucketSize)
                            {
                                bucketContentSize[hashInd]++;
                            }
                        } // Loop over all points in the reference set.
                    }
                    break;
                case 3:
                    // Obtain the weights for the second hash.
                    secondHashWeights = arma::floor(arma::randu(numShears) * (double) secondHashSize);
                    powerSize = pow(2, ceil(log2(numDimensions)));
                    shearsTable.clear();
                    for (size_t i = 0; i < numTables; i++)
                    {
                      arma::imat shearMat;
                      shearMat = arma::randi<arma::imat>(powerSize * 3, numShears, arma::distr_param(-1, +1));

                      shearsTable.push_back(shearMat);
                    }
                    
                    for(size_t i = 0; i < referenceSet->n_cols; i++)
                    {
                        arma::vec query = referenceSet->col(i);
                        arma::mat polytopeMat = hashTypeCrossPolytopeOnePoint(query, numTables);
                        //polytopeMat.raw_print(std::cout, "P =");
                        // Step VI: Putting the points in the 'secondHashTable' by hashing the key.
                        // Now we hash every key, point ID to its corresponding bucket.
                        arma::rowvec secondHashVec = secondHashWeights.t() * arma::floor(polytopeMat);
                        
                        // This gives us the bucket for the corresponding point ID.
                        for (size_t j = 0; j < secondHashVec.n_elem; j++)
                        {
                            secondHashVec[j] = (double) ((size_t) secondHashVec[j] % secondHashSize);
                        }
                        //Log::Assert(secondHashVec.n_elem == referenceSet->n_cols);

                        // Insert the point in the corresponding row to its bucket in the 'secondHashTable'.
                        for (size_t j = 0; j < secondHashVec.n_elem; j++) 
                        {
                            // This is the bucket number.
                            size_t hashInd = (size_t) secondHashVec[j];
                            // The point ID is 'j'.

                            // If this is currently an empty bucket, start a new row keep track of which row corresponds to the bucket.
                            if (bucketContentSize[hashInd] == 0) 
                            {
                                // Start a new row for hash.
                                bucketRowInHashTable[hashInd] = numRowsInTable;
                                secondHashTable(numRowsInTable, 0) = i;
                                numRowsInTable++;
                            }
                            else 
                            {
                                // If bucket is already present in the 'secondHashTable', find the corresponding row and insert the point ID in this row unless the bucket is full, in which case, do nothing.
                                if (bucketContentSize[hashInd] < bucketSize)
                                {
                                    secondHashTable(bucketRowInHashTable[hashInd], bucketContentSize[hashInd]) = i;
                                }
                            }

                            // Increment the count of the points in this bucket.
                            if (bucketContentSize[hashInd] < bucketSize)
                            {
                                bucketContentSize[hashInd]++;
                            }
                        } // Loop over all points in the reference set.
                    }
                    break;
            }
            // Step VII: Condensing the 'secondHashTable'.
            size_t maxBucketSize = 0;
            for (size_t i = 0; i < bucketContentSize.n_elem; i++) 
            {
                if (bucketContentSize[i] > maxBucketSize) 
                {
                    maxBucketSize = bucketContentSize[i];
                }
            }
            Log::Info << "Final hash table size: (" << numRowsInTable << " x " << maxBucketSize << ")" << std::endl;
            secondHashTable.resize(numRowsInTable, maxBucketSize);
        }       
        
        void hashModel::hashType2StableDistribution(size_t* numRowsInTable) 
        {            
            for (size_t i = 0; i < numTables; i++) 
            {
                // Step IV: Obtain the 'numProj' projections for each table.
                // For L2 metric, 2-stable distributions are used, and the normal Z ~ N(0, 1) is a 2-stable distribution.
                arma::mat projMat;
                projMat.randn(referenceSet->n_rows, numProj);

                // Save the projection matrix for querying.
                projections.push_back(projMat);

                // Step V: create the 'numProj'-dimensional key for each point in each table.
                // The following code performs the task of hashing each point to a 'numProj'-dimensional integer key.  Hence you get a ('numProj' x 'referenceSet.n_cols') key matrix.
                // For a single table, let the 'numProj' projections be denoted by 'proj_i' and the corresponding offset be 'offset_i'.
                // Then the key of a single point is obtained as: key = { floor( (<proj_i, point> + offset_i) / 'hashWidth' ) forall i }
                arma::mat offsetMat = arma::repmat(offsets.unsafe_col(i), 1,    referenceSet->n_cols);
                arma::mat hashMat = projMat.t() * (*referenceSet);
                hashMat += offsetMat;
                hashMat /= hashWidth;

                // Step VI: Putting the points in the 'secondHashTable' by hashing the key.
                // Now we hash every key, point ID to its corresponding bucket.
                arma::rowvec secondHashVec = secondHashWeights.t() * arma::floor(hashMat);

                // This gives us the bucket for the corresponding point ID.
                for (size_t j = 0; j < secondHashVec.n_elem; j++)
                {
                    secondHashVec[j] = (double) ((size_t) secondHashVec[j] % secondHashSize);
                }
                //Log::Assert(secondHashVec.n_elem == referenceSet->n_cols);

                // Insert the point in the corresponding row to its bucket in the 'secondHashTable'.
                for (size_t j = 0; j < secondHashVec.n_elem; j++) 
                {
                    // This is the bucket number.
                    size_t hashInd = (size_t) secondHashVec[j];
                    // The point ID is 'j'.

                    // If this is currently an empty bucket, start a new row keep track of which row corresponds to the bucket.
                    if (bucketContentSize[hashInd] == 0) 
                    {
                        // Start a new row for hash.
                        bucketRowInHashTable[hashInd] = *numRowsInTable;
                        secondHashTable(*numRowsInTable, 0) = j;
                        (*numRowsInTable)++;
                    }
                    else 
                    {
                        // If bucket is already present in the 'secondHashTable', find the corresponding row and insert the point ID in this row unless the bucket is full, in which case, do nothing.
                        if (bucketContentSize[hashInd] < bucketSize)
                        {
                            secondHashTable(bucketRowInHashTable[hashInd], bucketContentSize[hashInd]) = j;
                        }
                    }

                    // Increment the count of the points in this bucket.
                    if (bucketContentSize[hashInd] < bucketSize)
                    {
                        bucketContentSize[hashInd]++;
                    }
                } // Loop over all points in the reference set.
            } // Loop over tables.
        }
        
        template<typename VecType>    
        arma::mat hashModel::hashTypeHyperplaneOnePoint(const VecType& queryPoint, size_t numTablesToSearch) const
        {
            arma::mat allCutsInTables(numPlanes, numTablesToSearch);
            
            for (size_t i = 0; i < numTablesToSearch; i++)
            {
                arma::vec cut = planes[i].t() * (queryPoint); //after multiplication, the cut.size = numPlanes

                for (size_t j = 0; j < numPlanes; j++)
                {
                    cut[j] = cut[j] >= 0 ? 1 : 0;
                }
                allCutsInTables.unsafe_col(i) = cut;
            }
            return allCutsInTables;
        }
        
        template<typename VecType>
        void hashModel::hashQuery(const VecType& queryPoint, size_t numTablesToSearch, arma::Col<size_t>& refPointsConsidered) const
        {
            arma::rowvec hashVec;
            
            switch(hashType)
            {
                case 1:
                {
                    arma::mat allProjInTables(numProj, numTablesToSearch);
                    for (size_t i = 0; i < numTablesToSearch; i++)
                    {
                        allProjInTables.unsafe_col(i) = projections[i].t() * queryPoint;
                    }
                    allProjInTables += offsets.cols(0, numTablesToSearch - 1);
                    allProjInTables /= hashWidth;
                    // Compute the hash value of each key of the query into a bucket of the 'secondHashTable' using the 'secondHashWeights'.
                    hashVec = secondHashWeights.t() * arma::floor(allProjInTables);
                }
                    break;
                case 2:
                {
                    arma::mat allCutsInTables = hashTypeHyperplaneOnePoint(queryPoint, numTablesToSearch);
                    hashVec = secondHashWeights.t() * arma::floor(allCutsInTables);
                }
                    break;
                case 3:
                    arma::mat polytopeMat = hashTypeCrossPolytopeOnePoint(queryPoint, numTablesToSearch);
                    hashVec = secondHashWeights.t() * arma::floor(polytopeMat);
                    break;
            }
            
            for (size_t i = 0; i < hashVec.n_elem; i++)
            {
                hashVec[i] = (double) ((size_t) hashVec[i] % secondHashSize);
            }
            Log::Assert(hashVec.n_elem == numTablesToSearch);
            
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
        }
        
        template<typename VecType> 
        arma::mat hashModel::hashTypeCrossPolytopeOnePoint(const VecType& queryPoint, size_t numTablesToSearch) const
        {
            arma::mat allRotationsInTables(numShears, numTablesToSearch);
            for (size_t i = 0; i < numTablesToSearch; i++)
            {
                arma::vec rotation(numShears);
                rotation.zeros();

                for (size_t k = 0; k < numShears; k++)
                {
                    arma::vec aux;
                    aux.set_size(powerSize);
                    aux.zeros();

                    //loading in the query point
                    for (size_t j = 0; j < numDimensions; j++)
                    {
                        aux[j] = queryPoint[j];
                    }
                    //padding to make the vector size a power of 2
                    for (size_t j = numDimensions; j < powerSize; j++)
                    {
                        aux[j] = 0;
                    }
                    //performing pseudo-random rotation of the point.
                    for (size_t d = 0; d < 3; d++)
                    {
                        for (size_t j = 0; j < numDimensions; j++)
                        {
                            aux[j] = aux[j] * shearsTable[i](d * numDimensions + j, k);
                        }
                        HadamardTransform(aux);
                    }
                   
                    //finding index of closest point on the cross-polytope
                    size_t maxIndex = 0;
                    size_t maxMagnitude = 0;
                    for (size_t j = 0; j < powerSize; j++)
                    {
                        if(maxMagnitude < abs(aux[j]))
                        {
                            maxIndex = j;
                            maxMagnitude = abs(aux[j]);
                        }
                    }
                    if(aux[maxIndex] < 0) // k ?
                    {
                        maxIndex += powerSize;
                    }
                    rotation[k] = maxIndex;
                }
                allRotationsInTables.unsafe_col(i) = rotation;
            }
            return allRotationsInTables;
        }
        
        template<typename VecType> 
        void hashModel::HadamardTransform(VecType& query) const
        {
            VecType aux(query.size());
            aux.zeros();
            VecType res(query.size());
            res.zeros();
           
            //loading in the query point
            for(size_t i = 0; i < query.size(); i++)
            {
                aux[i] = query[i];
            }
            //iterating over every size of division
            for(size_t d = query.size(); d >= 2; d/=2)
            {
                res.zeros();
                //going through each subdivision
                for(size_t r = 0; r < query.size(); r += d)
                {
                    for(size_t i = 0; i < d/2; i++)
                    {
                        res[r + i]       += aux[r + i];
                        res[r + i]       += aux[r + d/2 + i];
                        res[r + d/2 + i] += aux[r + i];
                        res[r + d/2 + i] -= aux[r + d/2 + i];
                    }
                }
                //swapping to prepare the next auxiliary
                VecType temp = res;
                res = aux;
                aux = temp;
            }
            query = aux;
        }
        
        template<typename VecType> 
        double hashModel::cosineDistance(const VecType& A, const VecType& B) const
        {
            return acos(arma::norm_dot(A,B));
        } 
        
        template<typename VecType> 
        double hashModel::angularDistance(const VecType& A, const VecType& B) const
        {
            return arma::norm_dot(A,B);
        }
                
        template<typename Archive>       
        void hashModel::Serialize(Archive& ar, const unsigned int /* version */) 
        {
            using data::CreateNVP;

            ar & CreateNVP(referenceSet, "referenceSet");
            ar & CreateNVP(hashType, "hashType");
            ar & CreateNVP(secondHashSize, "secondHashSize");           
            ar & CreateNVP(bucketSize, "bucketSize");
            ar & CreateNVP(secondHashWeights, "secondHashWeights");
            ar & CreateNVP(secondHashTable, "secondHashTable");
            ar & CreateNVP(bucketContentSize, "bucketContentSize");
            ar & CreateNVP(bucketRowInHashTable, "bucketRowInHashTable");
  
            ar & CreateNVP(numProj, "numProj");
            ar & CreateNVP(numTables, "numTables");
            ar & CreateNVP(hashWidth, "hashWidth");
            if (Archive::is_loading::value)
            {
                projections.clear();
            }          
            ar & CreateNVP(projections, "projections");
            ar & CreateNVP(offsets, "offsets");
            
            ar & CreateNVP(numDimensions, "numDimensions");
            ar & CreateNVP(numPlanes, "numPlanes");
            ar & CreateNVP(planes, "planes");
            
            ar & CreateNVP(numShears, "numShears");
            ar & CreateNVP(shearsTable, "shearsTable");
        }
    }
}
#endif /* HASH_MODEL_IMPL_HPP */