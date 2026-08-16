use anyhow::Result;

/// Liquid Neural Network / NCP (Neural Circuit Policy)
/// Closed-form continuous-time (CfC) cell for temporal processing
pub struct LiquidCell {
    hidden_size: usize,
    input_size: usize,
    // NCP wiring matrix: maps input → interneurons → motor neurons
    wiring: WiringMatrix,
    state: Vec<f32>,
}

struct WiringMatrix {
    sensory_to_inter: Vec<Vec<f32>>,
    inter_to_motor: Vec<Vec<f32>>,
    // Liquid time-constant per neuron (learned)
    tau: Vec<f32>,
}

impl LiquidCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let wiring = WiringMatrix {
            sensory_to_inter: vec![vec![0.0; hidden_size]; input_size],
            inter_to_motor: vec![vec![0.0; hidden_size]; hidden_size],
            tau: vec![1.0; hidden_size],
        };
        Self {
            hidden_size,
            input_size,
            wiring,
            state: vec![0.0; hidden_size],
        }
    }

    pub fn forward(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        // Closed-form continuous-time (CfC) ODE solve
        // state' = -state / tau + W_input * input + bias
        let mut new_state = self.state.clone();
        for i in 0..self.hidden_size {
            let mut sum = 0.0;
            for j in 0..self.input_size.min(input.len()) {
                sum += self.wiring.sensory_to_inter[j][i] * input[j];
            }
            // Liquid time constant gating
            let decay = (-1.0 / self.wiring.tau[i]).exp();
            new_state[i] = decay * self.state[i] + (1.0 - decay) * sum.tanh();
        }
        Ok(new_state)
    }

    pub fn reset(&mut self) {
        self.state = vec![0.0; self.hidden_size];
    }
}

/// NCP wired neural circuit
pub struct NCP {
    liquid: LiquidCell,
    motor: Vec<f32>,
}

impl NCP {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            liquid: LiquidCell::new(input_size, hidden_size),
            motor: vec![0.0; output_size],
        }
    }

    pub fn step(&mut self, input: Vec<f32>) -> Result<Vec<f32>> {
        let hidden = self.liquid.forward(input)?;
        // Motor neuron readout
        let output = hidden.iter().take(self.motor.len()).cloned().collect();
        Ok(output)
    }
}