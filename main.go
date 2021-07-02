package main

import (
	"time"

	log "github.com/sirupsen/logrus"

	"code.ia/internal/config"
	"code.ia/internal/neuralnets"
)

func main() {
	if err := config.ParseSettings(); err != nil {
		log.Fatal(err)
	}

	settings := &neuralnets.NeuralNetworkSettings{
		InputNeurons:  4,
		OutputNeurons: 3,
		HiddenNeurons: 4,
		Epochs:        1000,
		LearningRate:  0.01,
	}

	start := time.Now()

	network := neuralnets.NewNetwork(*settings)
	if err := network.Train(config.Features, config.Labels); err != nil {
		log.Fatal(err)
	}

	elapsed := time.Since(start)

	network.Print()

	accuracy, err := network.Validate()
	if err != nil {
		log.Fatal(err)
	}
	log.Infof("\nTasa de aciertos = %0.4f\n\n", accuracy)

	log.Infof("\n Tiempo transcurrido: %v", elapsed)
}
