/****************************************************************************

    ePMC - an extensible probabilistic model checker
    Copyright (C) 2017

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*****************************************************************************/

#include <stdlib.h>
#include <math.h>
#include "epmc_error.h"

__attribute__ ((visibility("default")))
epmc_error_t double_dtmc_unbounded_cumulative_jacobi(int relative,
        double precision, int numStates, int *stateBounds, int *targets,
        double *weights, double *values, double *cumul, int *iterationsResult,
        volatile int *numIterationsFeedback, volatile double *differenceFeedback) {
    double *presValues = values;
    double *nextValues = malloc(sizeof(double) * numStates);
    int iterations = 0;
    if (nextValues == NULL) {
        return OUT_OF_MEMORY;
    }
    double *allocated = nextValues;
    for (int state = 0; state < numStates; state++) {
        nextValues[state] = 0.0;
    }
    double maxDiff;
    do {
        maxDiff = 0.0;
        for (int state = 0; state < numStates; state++) {
            int from = stateBounds[state];
            int to = stateBounds[state + 1];
            double nextStateProb = cumul[state];
            for (int succ = from; succ < to; succ++) {
                double weight = weights[succ];
                int succState = targets[succ];
                double succStateProb = presValues[succState];
                nextStateProb += weight * succStateProb;
            }
            double diff = fabs(nextStateProb - presValues[state]);
            if (relative && presValues[state] != 0.0) {
                diff /= presValues[state];
            }
            maxDiff = diff > maxDiff ? diff : maxDiff;
            nextValues[state] = nextStateProb;
        }
        double *swap = presValues;
        presValues = nextValues;
        nextValues = swap;
        *numIterationsFeedback = iterations;
        *differenceFeedback = maxDiff;
        iterations++;
    } while (maxDiff > precision / 2);
    for (int state = 0; state < numStates; state++) {
        values[state] = presValues[state];
    }
    free(allocated);
    iterationsResult[0] = iterations;
    return SUCCESS;
}
