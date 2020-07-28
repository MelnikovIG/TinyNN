using Xunit;

namespace TinyNN.Tests
{
    public class SerializationTests
    {
        [Fact]
        public void SaveLoadWeightsTest()
        {
            Layer[] layers = {
                new Layer(5, Activations.Sigmoid),
                new Layer(10, Activations.Sigmoid),
                new Layer(5, Activations.Sigmoid),
                new Layer(1, Activations.Sigmoid),
            };
            
            var network = new Network(layers);

            var serializedWeights = network.CloneWeights();
            network.SetWeights(serializedWeights, true);
        }
    }
}