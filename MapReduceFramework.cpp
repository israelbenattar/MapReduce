#include "MapReduceFramework.h"
#include "Barrier.h"
#include <pthread.h>
#include <atomic>
#include <semaphore.h>
#include <algorithm>
#include <iostream>
#include <cmath>

struct ThreadContext;


/**
 * the JobContext struct wiche contain a job context.
 */
typedef struct
{
    //num of multi thread level.
	const int numOfThreads;

	//array of threads.
	pthread_t *threads;

	//array of pointer to the threads contexts.
	ThreadContext **contexts;

	//the client object.
	const MapReduceClient *client;

	//vector of input elements.
	const InputVec *inputVec;

	//vector of intermediate vectors.
	std::vector<IntermediateVec *> *intermediateVec;

	//the output vector.
	OutputVec *outputVec;

	//barrier object.
	Barrier *barrier;

	//the job semaporh.
	sem_t sem;

	//the job mutexes.
	pthread_mutex_t mutex;
	pthread_mutex_t waitForMutex;

	//all the atomic of the job
	std::atomic<int> mapAtomicCounter;
	std::atomic<uint64_t> stateAtomic;
	std::atomic<int> interElemAtomic;
	std::atomic<bool> jobEnded;
	std::atomic<bool> startState;

} JobContext;


/**
 * the ThreadContext struct wiche contain a thread context.
 */
struct ThreadContext
{
    //the id of the thread.
	int id;

	//a vector of the thread intermediate element.
	IntermediateVec *intermediateVec;

	//the context of the job that the thread is doing.
	JobContext *jobHandler;
};

void shuffle(JobContext *jobContext);


void *run(void *threadContext);

/**
 * The function saves the intermediary element in the context data structures. In
 * addition, the function updates the number of intermediary elements using atomic counter.
 * @param key - The key of the intermediate element.
 * @param value - The value of the intermediate element.
 * @param context - The thread context of the current thread.
 */
void emit2(K2 *key, V2 *value, void *context)
{
	auto *context1 = (ThreadContext *) context;
	context1->intermediateVec->push_back({key, value});
	(context1->jobHandler->interElemAtomic)++;
}

/**
 * The function saves the output element in the context data structures(output vector).
 * In addition, the function updates the number of output elements using atomic counter
 * @param key - The key of the intermediate element.
 * @param value - The value of the intermediate element.
 * @param context - The job context of the current job.
 */
void emit3(K3 *key, V3 *value, void *context)
{
	auto *context1 = (JobContext *) context;
    if (pthread_mutex_lock(&context1->mutex) != 0)
    {
        std::cerr << "system error: pthread_mutex_lock filed." << std::endl;
        exit(1);
    }
	context1->outputVec->push_back({key, value});
	(context1->stateAtomic)++;
    if (pthread_mutex_unlock(&context1->mutex) != 0)
    {
        std::cerr << "system error: pthread_mutex_unlock filed." << std::endl;
        exit(1);
    }
}


/**
 * The function initilaiz the job context.
 * @param multiThreadLevel - the number of worker threads to be used for running the algorithm.
 * @param inputVec - A vector of type std::vector<std::pair<K1*, V1*>>, the input elements.
 * @param client - The implementation of MapReduceClient.
 * @param outputVec - a vector of type std::vector<std::pair<K3*, V3*>>, to which the output elements will be added.
 * @return - A pointer to the created job context.
 */
JobContext *
init_job(const int multiThreadLevel, const InputVec &inputVec, const MapReduceClient &client,
		OutputVec
&outputVec)
{
	auto *jobHandle = new JobContext{(const int) multiThreadLevel};
	jobHandle->threads = new pthread_t[multiThreadLevel];
	jobHandle->contexts = new ThreadContext *[multiThreadLevel];
	jobHandle->client = &client;
	jobHandle->inputVec = &inputVec;
	jobHandle->intermediateVec = new std::vector<IntermediateVec *>;
	jobHandle->outputVec = &outputVec;
	jobHandle->barrier = new Barrier(multiThreadLevel);
	jobHandle->mapAtomicCounter = 0;
	jobHandle->interElemAtomic = 0;
	jobHandle->jobEnded = false;
	jobHandle->startState = false;
	uint64_t inSize = inputVec.size();
	jobHandle->stateAtomic = (inSize << 31);
    if (sem_init(&(jobHandle->sem), 0, 0) != 0)
    {
        std::cerr << "system error: sem_init filed." << std::endl;
        exit(1);
    }
    if (pthread_mutex_init(&(jobHandle->mutex), nullptr) != 0 ||
        pthread_mutex_init(&(jobHandle->waitForMutex), nullptr) != 0)
    {
        std::cerr << "system error: pthread_mutex_init filed." << std::endl;
        exit(1);
    }
	return jobHandle;
}

/**
 * This function starts running the MapReduce algorithm (with several threads) and returns a JobHandle.
 * @param client - The implementation of MapReduceClient.
 * @param inputVec - A vector of type std::vector<std::pair<K1*, V1*>>, the input elements.
 * @param outputVec - A vector of type std::vector<std::pair<K3*, V3*>>, to which the output elements will be added.
 * @param multiThreadLevel - The number of worker threads to be used for running the algorithm.
 * @return - The function returns JobHandle that will be used for monitoring the job.
 */
JobHandle startMapReduceJob(const MapReduceClient &client,
							const InputVec &inputVec, OutputVec &outputVec,
							int multiThreadLevel)
{
	auto *jobHandle = init_job(multiThreadLevel, inputVec, client, outputVec);

	for (int i = 0; i < multiThreadLevel; i++)
	{
		auto *threadContext = new ThreadContext{i,
												new IntermediateVec(),
												jobHandle};
		jobHandle->contexts[i] = threadContext;

		if (pthread_create(&jobHandle->threads[i], nullptr, &run, threadContext) != 0)
		{
			std::cerr << "system error: pthread_create filed." << std::endl;
			exit(1);
		}

	}

	return (JobHandle) jobHandle;

}

/**
 * This function gets JobHandle returned by startMapReduceFramework and waits until it is finished.
 * @param job - the JobHandle
 */
void waitForJob(JobHandle job)
{
	auto *job1 = (JobContext *) job;
	pthread_mutex_lock(&(job1->waitForMutex));
	if (job1->jobEnded)
	{
		pthread_mutex_unlock(&(job1->waitForMutex));
		return;
	}
	pthread_t *threads = job1->threads;
	for (int i = 0; i < job1->numOfThreads; i++)
	{
		if (pthread_join(threads[i], nullptr) != 0)
		{
			std::cerr << "system error: pthread_join filed." << std::endl;
			exit(1);
		}
	}
	job1->jobEnded = true;
	pthread_mutex_unlock(&(job1->waitForMutex));
}

/**
 * This function gets a JobHandle and updates the state of the job into the givenJobState struct.
 * @param job - the JobHandle
 * @param state - the state
 */
void getJobState(JobHandle job, JobState *state)
{
	auto *jobC = (JobContext *) job;
    float percentage;
    uint64_t  f = jobC->stateAtomic.load();
    uint64_t a = ((f >> 31) & (0x7fffffff));
    uint64_t b = ((f) & (0x7fffffff));
    stage_t stage = static_cast<stage_t>(f>> 62);
    if (a == 0)
    {
        percentage = 0;
    } else {
        percentage = std::floor((100 * b) / a);
    }
	state->stage = stage;
    state->percentage = percentage;



}

/**
 * this function releasing all resources of a job.
 * @param job - JobHandle.
 */
void closeJobHandle(JobHandle job)
{
	auto *jobC = (JobContext *) job;
	if (!(jobC->jobEnded))
	{
		waitForJob(job);
	}
	if (sem_destroy(&(jobC->sem)) != 0)
	{
		std::cerr << "system error: pthread_mutex_destroy filed." << std::endl;
		exit(1);
	}
	if (pthread_mutex_destroy(&(jobC->mutex)) != 0)
	{
		std::cerr << "system error: pthread_mutex_destroy filed." << std::endl;
		exit(1);
	}
	for (int i = 0; i < jobC->numOfThreads; i++)
	{
		delete jobC->contexts[i]->intermediateVec;
		delete jobC->contexts[i];
		jobC->contexts[i] = nullptr;
	}
	delete[]jobC->contexts;
	delete[]jobC->threads;
	int s = jobC->intermediateVec->size();
	for (int i = 0; i < s; i++)
	{
		delete jobC->intermediateVec->at(i);
	}
	delete jobC->intermediateVec;
	delete jobC->barrier;
	delete jobC;
	jobC = nullptr;
}

/**
 * In this function each thread reads pairs of (k1,v1) from the input vector and calls the map
 * function on each of them.
 * @param threadContext - The context of the current thread.
 */
void map_phase(ThreadContext *threadContext)
{
    uint64_t h;
    pthread_mutex_lock(&threadContext->jobHandler->mutex);
    if (!(threadContext->jobHandler->startState))
    {
        h = MAP_STAGE;
        threadContext->jobHandler->stateAtomic += (h << 62);
        threadContext->jobHandler->startState = true;

    }
    pthread_mutex_unlock(&threadContext->jobHandler->mutex);

	int job_id = (threadContext->jobHandler->mapAtomicCounter)++;

	while (job_id < threadContext->jobHandler->inputVec->size())
	{
		const InputPair job = threadContext->jobHandler->inputVec->at(job_id);
		threadContext->jobHandler->client->map(job.first, job.second, threadContext);
		threadContext->jobHandler->stateAtomic++;
		job_id = (threadContext->jobHandler->mapAtomicCounter)++;
	}
}

/**
 * In this function each thread will use sort to sort its intermediate vector according o the keys.
 * @param threadContext - The context of the current thread.
 */
void sort_phase(ThreadContext *threadContext)
{
	std::sort(threadContext->intermediateVec->begin(), threadContext->intermediateVec->end(),
			  [](IntermediatePair p1, IntermediatePair p2) { return *p1.first < *p2.first; });
}

/**
 * This function create new sequences of (k2,v2)where in each sequence all keys are identical and all
 * elements with a given key are in a single sequence.
 * @param threadContext - The context of the shuffling thread.
 */
void shuffle_phase(ThreadContext *threadContext)
{
    uint64_t h;
    threadContext->jobHandler->mapAtomicCounter = 0;
    threadContext->jobHandler->startState = false;
    h = SHUFFLE_STAGE;
    uint64_t i = threadContext->jobHandler->interElemAtomic.load();
    threadContext->jobHandler->stateAtomic = ((h << 62) + (i << 31));

	shuffle(threadContext->jobHandler);

	if (sem_post(&(threadContext->jobHandler->sem)) != 0)
	{
		std::cerr << "system error: sem_post filed." << std::endl;
		exit(1);
	}
}

/**
 * This function in turn will produce (k3,v3) pairs and will call emit3 to add them to the framework datastructures.
 * @param threadContext - The context of the current thread.
 */
void reduce_phase(ThreadContext *threadContext)
{
    uint64_t h;
    pthread_mutex_lock(&threadContext->jobHandler->mutex);
    if (!(threadContext->jobHandler->startState))
    {

        h = REDUCE_STAGE;
        threadContext->jobHandler->stateAtomic = (h << 62);
        threadContext->jobHandler->stateAtomic += (threadContext->jobHandler->intermediateVec->size() << 31);
        threadContext->jobHandler->startState = true;

    }
    pthread_mutex_unlock(&threadContext->jobHandler->mutex);
	int job_id = threadContext->jobHandler->mapAtomicCounter++;
	while (job_id < threadContext->jobHandler->intermediateVec->size())
	{

		const IntermediateVec *job = threadContext->jobHandler->intermediateVec->at(job_id);


		threadContext->jobHandler->client->reduce(job, threadContext->jobHandler);

		job_id = threadContext->jobHandler->mapAtomicCounter++;
	}
}

/**
 * This function run the MapReduce algorithm with.
 * @param threadContextIn - A thread context.
 * @return - null
 */
void *run(void *threadContextIn)
{
	auto *threadContext = (ThreadContext *) threadContextIn;

	map_phase(threadContext);

	sort_phase(threadContext);


	threadContext->jobHandler->barrier->barrier();


	if (threadContext->id == 0)
	{

		shuffle_phase(threadContext);
	}

	if (sem_wait(&(threadContext->jobHandler->sem)) != 0)
	{
		std::cerr << "system error: sem_wait filed." << std::endl;
		exit(1);
	}
	if (sem_post(&(threadContext->jobHandler->sem)) != 0)
	{
		std::cerr << "system error: sem_post filed." << std::endl;
		exit(1);
	}

	reduce_phase(threadContext);

    return nullptr;

}

/**
 * This function find the max key and return it. in addition the function return a vector withe the
 * indexes where the mex element are.
 * @param context - An array of thread contexts.
 * @param numOfThreads - Num of running threads.
 * @return - The max key and a vector of indexes or null if all the context vector are empty.
 */
std::pair<K2 *, std::vector<int>> max_key(ThreadContext **context, int numOfThreads)
{
	std::vector<int> maxVec;
	K2 *max = nullptr;
	for (int i = 0; i < numOfThreads; i++)
	{
		if (context[i]->intermediateVec->empty())
		{
			continue;
		}
		if (max == nullptr)
		{
			max = context[i]->intermediateVec->back().first;
			maxVec.push_back(i);
			continue;
		}
		IntermediatePair maxK = context[i]->intermediateVec->back();
		if (!((*max < *maxK.first) || (*maxK.first < *max)))
		{
			maxVec.push_back(i);
			continue;
		}
		if (*max < *maxK.first)
		{
			max = maxK.first;
			maxVec.clear();
			maxVec.push_back(i);
			continue;
		}
	}
	return {max, maxVec};
}

/**
 * this function making the actuall shuffle part
 * @param jobContext - the job that we shuffle
 */
void shuffle(JobContext *jobContext)
{
	int numThreads = jobContext->numOfThreads;
	ThreadContext **pContext = jobContext->contexts;

	K2 *maxKey;
	std::vector<int> maxVec;


	std::pair<K2 *, std::vector<int>> max = max_key(pContext, numThreads);
	maxKey = max.first;
	maxVec = max.second;

	while (maxKey)
	{
		auto *currentVec = new IntermediateVec();
		for (int &it : maxVec)
		{
			IntermediateVec *currentThreadVec = pContext[it]->intermediateVec;

			while (!currentThreadVec->empty() && !(*maxKey < *currentThreadVec->back().first)
				   && !(*currentThreadVec->back().first < *maxKey))
			{
				jobContext->stateAtomic++;
				currentVec->push_back(currentThreadVec->back());
				currentThreadVec->pop_back();
			}
		}
		jobContext->intermediateVec->push_back(currentVec);
		max = max_key(pContext, numThreads);
		maxKey = max.first;
		maxVec = max.second;
	}
}
