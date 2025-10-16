import numpy as np
cimport numpy as np
from bisect import bisect_left,bisect_right
cimport cython


cdef double m_cv_two(double t1, double t2, double t3):
    cdef double i1 = t2 -t1
    cdef double i2 =  t3-t2
    return 2 * abs(i2-i1)/(i2+i1)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def time_resolved_cv_two(np.ndarray[np.float64_t, ndim=2]  spiketimes,int window,object tlim = None,int min_vals = 10,int tstep = 1):
    
    if tlim is None:
        tlim = [np.nanmin(spiketimes[0]),np.nanmax(spiketimes[0])]
    tlim =[float(t) for t in tlim]
    cdef np.ndarray[np.int64_t,ndim=1] order
    
    spiketimes = spiketimes[:,np.isfinite(spiketimes[0])]
    order = np.argsort(spiketimes[0])
    spiketimes = spiketimes[:,order]
    order = np.argsort(spiketimes[1],kind = 'mergesort')
    spiketimes = spiketimes[:,order]
    
    
    cdef double tmax=tlim[1]
    cdef int total_length = spiketimes.shape[1]
    cdef int result_length = (tlim[1]-tlim[0])/tstep - window
    cdef np.ndarray[np.float64_t,ndim = 1] time = np.zeros((result_length))
    cdef np.ndarray[np.float64_t,ndim = 1] window_start_times = np.arange(0,result_length,tstep)+tlim[0]
    cdef np.ndarray[np.float64_t,ndim = 1] window_end_times = window_start_times+window
    cdef np.ndarray[np.float64_t,ndim = 1] ms = np.zeros((result_length))
    cdef np.ndarray[np.float64_t,ndim = 1] counts = np.zeros((result_length))
    cdef np.ndarray[np.float64_t,ndim = 1] cv2s = np.zeros((result_length))

    cdef int window_start = tlim[0]
    cdef int window_end = tlim[0] + window
    cdef int start_index = 0
    cdef int index
    cdef int current_window = 0
    cdef current_trial = 0
    cdef double time1,time2,time3,trial1,trial3
    
    for index in range(start_index,total_length-2):

        time1 = spiketimes[0,index]
        time2 = spiketimes[0,index+1]
        time3 = spiketimes[0,index+2]
        trial1 = spiketimes[1,index]
        trial3 = spiketimes[1,index+2]
        
        if time3>tmax:
            #reached end of tlim
            current_window = 0
            continue
        
        if trial1!=trial3:
            # all spikes must lie in same trial
            current_window = 0
            continue
        
        if (time3 - time1) > window:
            # all three spikes must lie in same window
            continue
        for current_window in range(result_length):
            if time1 < window_start_times[current_window]:
                continue
            if time3>= window_end_times[current_window]:
                continue
            ms[current_window] += m_cv_two(time1,time2,time3)
            counts[current_window] += 1
    
    time = 0.5*(window_start_times+window_end_times)
    #print window_start_times
    #print window_end_times
    cv2s[counts>0] = ms[counts>0]/counts[counts>0]
    cv2s[counts<min_vals] = np.nan

    return cv2s,time

        

        

