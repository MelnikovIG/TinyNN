using System.Text.Json.Serialization;
using Newtonsoft.Json;
using JsonConverter = System.Text.Json.Serialization.JsonConverter;

namespace TinyNN
{
    public partial class Network
    {
        public double[][][] CloneWeights()
        {
            var serialized = SerializeWeights();
            return JsonConvert.DeserializeObject<double[][][]>(serialized);
        }

        public string SerializeWeights()
        {
            return JsonConvert.SerializeObject(_weights);
        }

        public void SetWeights(double[][][] weights, bool needClone)
        {
            if (needClone)
            {
                var serialized = JsonConvert.SerializeObject(weights);
                weights = JsonConvert.DeserializeObject<double[][][]>(serialized);
            }

            _weights = weights;
        }
    }
}