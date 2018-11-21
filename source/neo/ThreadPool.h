// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <vector>
#include <list>
#include <memory>

namespace ogmaneo {
	/*!
	\brief Work item. Inherit from this to be able to add to thread pool
	*/
	class WorkItem {
	private:
		std::atomic_bool _done;
		
	public:
		WorkItem() {
			_done = false;
		}

		virtual ~WorkItem() {}

		/*!
		\brief Override function to be run
		*/
		virtual void run() = 0;

		/*!
		\brief Whether the done flag has been set (for task completion)
		*/
		bool isDone() const {
			return _done;
		}

		friend class ThreadPool;
		friend class WorkerThread;
	};
	
	/*!
	\brief Worker thread. For internal use
	*/
	class WorkerThread {
    private:
        std::mutex _mutex;

	public:
		std::shared_ptr<std::thread> _thread;
		std::condition_variable _conditionVariable;

		bool _proceed;

		std::shared_ptr<class WorkItem> _item;

		class ThreadPool* _pPool;
		size_t _workerIndex;

		WorkerThread()
			: _pPool(nullptr), _workerIndex(0), _proceed(false)
		{}

		void start() {
			_thread = std::make_shared<std::thread>(&WorkerThread::run, this);
		}

		static void run(WorkerThread* pWorker);
		
		friend class ThreadPool;
	};
	
	/*!
	\brief Thread pool, using C++11 threading
	*/
	class ThreadPool {
	private:
		std::mutex _mutex;

		std::vector<std::unique_ptr<class WorkerThread>> _workers;

		std::list<size_t> _availableThreadIndicies;

		std::list<std::shared_ptr<class WorkItem>> _itemQueue;

	public:
		~ThreadPool() {
			destroy();
		}

		/*!
		\brief Create the pool
		\param numWorkers number of worker threads
		*/
		void create(size_t numWorkers);

		/*!
		\brief Destroy the thread pool
		*/
		void destroy();

		/*!
		\brief Add a work item
		\param item the work item to be enqueued
		*/
		void addItem(const std::shared_ptr<class WorkItem> &item);

		/*!
		\brief Get total number of worker threads
		*/
		size_t getNumWorkers() const {
			return _workers.size();
		}

		/*!
		\brief Wait for all items to be processed
		*/
		void wait();
		
		friend class WorkerThread;
	};
}
