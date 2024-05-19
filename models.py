import random

from causy.sample_generator import random_normal, TimeAwareNodeReference

from causy.sample_generator import SampleEdge
from data_generator import TimeseriesSampleGenerator

MODEL_ADMG = TimeseriesSampleGenerator(
        edges=[
            SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.9),
            SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Y"), 5),
            SampleEdge(TimeAwareNodeReference("Z", -2), TimeAwareNodeReference("Y"), 1),
            SampleEdge(TimeAwareNodeReference("Z", -1), TimeAwareNodeReference("Y"), 1),
        ],


)

MODEL_DAG_THREE_VARIABLES = TimeseriesSampleGenerator(
    edges=[
        SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.9),
        SampleEdge(TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Y"), 0.9),
        SampleEdge(TimeAwareNodeReference("Z", -1), TimeAwareNodeReference("Z"), 0.9),
        SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Y"), 5),
        SampleEdge(TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Z"), 7),
    ],

)

MODEL_DAG_TWO_VARIABLES = TimeseriesSampleGenerator(
    edges=[
        SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.9),
        SampleEdge(TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Y"), 0.9),
        SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Y"), 5),
    ],

)

MODEL_DAG_WITH_SMALL_AUTOCORR = TimeseriesSampleGenerator(
    edges=[
        SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("X"), 0.3),
        SampleEdge(TimeAwareNodeReference("Y", -1), TimeAwareNodeReference("Y"), 0.3),
        SampleEdge(TimeAwareNodeReference("X", -1), TimeAwareNodeReference("Y"), 1),
    ],
)
