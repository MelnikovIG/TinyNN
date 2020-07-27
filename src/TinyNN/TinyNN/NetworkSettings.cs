using System;

namespace TinyNN
{
    public class NetworkSettings
    {
        public int InputsCount;
        public int OutputsCount;
        public int HiddenLayersCount;
        public int TestInputsCount;
        public int TrainingExamplesCount;
        public int TrainingIterations;

        public int[] HiddenLayers;
        public InputData[] TestInputs;
        
        public InputData[] TrainingInputs;
        public OutputData[] ExpectedOutputs;
    }
}