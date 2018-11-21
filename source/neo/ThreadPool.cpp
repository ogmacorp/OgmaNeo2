// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ThreadPool.h"

#include <assert.h>
#include <iostream>

using namespace ogmaneo;

void WorkerThread::run(WorkerThread* pWorker) {
	while (true) {
		std::unique_lock<std::mutex> lock(pWorker->_mutex);

		pWorker->_conditionVariable.wait(lock, [pWorker] { return pWorker->_proceed; });

		if (pWorker->_pPool == nullptr) {
			lock.unlock();
			pWorker->_conditionVariable.notify_one();
			break;
		}

		while (pWorker->_proceed) {
			assert(pWorker->_item != nullptr);
			
			pWorker->_item->run();
			pWorker->_item->_done = true;

			ThreadPool* pPool = pWorker->_pPool;

			std::lock_guard<std::mutex> lock(pPool->_mutex);

			if (pPool->_itemQueue.empty()) {
				pPool->_availableThreadIndicies.push_back(pWorker->_workerIndex);
				pWorker->_proceed = false;
			}
			else {
				// Assign new task
				pWorker->_item = pPool->_itemQueue.front();
				pPool->_itemQueue.pop_front();
			}
		}

		lock.unlock();
		pWorker->_conditionVariable.notify_one();
	}
}

void ThreadPool::create(size_t numWorkers) {
	_workers.resize(numWorkers);

	// Add all threads as available and launch threads
	for (size_t i = 0; i < _workers.size(); i++) {
		_workers[i].reset(new WorkerThread());

		_availableThreadIndicies.push_back(i);

		// Block all threads as there are no tasks yet
		_workers[i]->_pPool = this;
		_workers[i]->_workerIndex = i;

		_workers[i]->start();
	}
}

void ThreadPool::destroy() {
	std::lock_guard<std::mutex> lock(_mutex);

	_itemQueue.clear();
	_availableThreadIndicies.clear();

	for (size_t i = 0; i < _workers.size(); i++) {
		std::unique_lock<std::mutex> lock(_workers[i]->_mutex);
		
		_workers[i]->_item = nullptr;
		_workers[i]->_pPool = nullptr;
		_workers[i]->_proceed = true;

		lock.unlock();

		_workers[i]->_conditionVariable.notify_one();

		_workers[i]->_thread->join();
	}
}

void ThreadPool::addItem(const std::shared_ptr<WorkItem> &item) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (!_availableThreadIndicies.empty()) {
		size_t workerIndex = _availableThreadIndicies.front();

		_availableThreadIndicies.pop_front();

		std::unique_lock<std::mutex> lock(_workers[workerIndex]->_mutex);

		_workers[workerIndex]->_item = item;
		_workers[workerIndex]->_proceed = true;

		lock.unlock();
		
		_workers[workerIndex]->_conditionVariable.notify_one();
	}
	else
		_itemQueue.push_back(item);
}

void ThreadPool::wait() {
	// Try to aquire every mutex until no tasks are left
	for (size_t i = 0; i < _workers.size(); i++) {
		std::unique_lock<std::mutex> lock(_workers[i]->_mutex);

		WorkerThread* pWorker = _workers[i].get();

		_workers[i]->_conditionVariable.wait(lock, [pWorker] { return !pWorker->_proceed; });
	}
}