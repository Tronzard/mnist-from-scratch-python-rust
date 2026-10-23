//! MNIST MLP using only the Rust standard library.
//! No tch, ndarray, autograd, neural network frameworks, or even RNG crates.

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct XorShift32(u32);
impl XorShift32 {
    fn new(seed: u32) -> Self { Self(if seed == 0 { 1 } else { seed }) }
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn uniform(&mut self, bound: f32) -> f32 {
        (2.0 * (self.next() as f32 / u32::MAX as f32) - 1.0) * bound
    }
    fn shuffle(&mut self, values: &mut [usize]) {
        for i in (1..values.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            values.swap(i, j);
        }
    }
}

struct Dataset {
    images: Vec<u8>,
    labels: Vec<u8>,
    count: usize,
}
impl Dataset {
    fn read(folder: &Path, prefix: &str) -> Result<Self, Box<dyn Error>> {
        let image_path = folder.join(format!("{prefix}-images-idx3-ubyte"));
        let label_path = folder.join(format!("{prefix}-labels-idx1-ubyte"));
        let raw_images = fs::read(&image_path)?;
        let raw_labels = fs::read(&label_path)?;
        if raw_images.len() < 16 || raw_labels.len() < 8 {
            return Err("IDX files are too small".into());
        }
        let u32be = |b: &[u8]| u32::from_be_bytes(b.try_into().unwrap()) as usize;
        let image_magic = u32be(&raw_images[0..4]);
        let label_magic = u32be(&raw_labels[0..4]);
        let count = u32be(&raw_images[4..8]);
        let label_count = u32be(&raw_labels[4..8]);
        let rows = u32be(&raw_images[8..12]);
        let cols = u32be(&raw_images[12..16]);
        if image_magic != 2051 || label_magic != 2049 || count != label_count
            || rows != 28 || cols != 28
            || raw_images.len() != 16 + count * 784 || raw_labels.len() != 8 + count {
            return Err("Invalid MNIST IDX headers, dimensions, or file lengths".into());
        }
        let labels = raw_labels[8..].to_vec();
        if labels.iter().any(|&label| label > 9) {
            return Err("MNIST labels must be in 0..9".into());
        }
        Ok(Self { images: raw_images[16..].to_vec(), labels, count })
    }
    fn image(&self, i: usize) -> &[u8] { &self.images[i * 784..(i + 1) * 784] }
}

struct Network {
    input: usize,
    hidden: usize,
    output: usize,
    w1: Vec<f32>, // row-major [input, hidden]
    b1: Vec<f32>,
    w2: Vec<f32>, // row-major [hidden, output]
    b2: Vec<f32>,
}
struct Gradients {
    w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>,
}
impl Gradients {
    fn new(m: &Network) -> Self {
        Self { w1: vec![0.0; m.w1.len()], b1: vec![0.0; m.b1.len()],
               w2: vec![0.0; m.w2.len()], b2: vec![0.0; m.b2.len()] }
    }
    fn clear(&mut self) {
        self.w1.fill(0.0); self.b1.fill(0.0); self.w2.fill(0.0); self.b2.fill(0.0);
    }
}
impl Network {
    fn new(input: usize, hidden: usize, output: usize, rng: &mut XorShift32) -> Self {
        let b1_bound = (6.0_f32 / input as f32).sqrt();
        let b2_bound = (6.0_f32 / (hidden + output) as f32).sqrt();
        Self {
            input, hidden, output,
            w1: (0..input * hidden).map(|_| rng.uniform(b1_bound)).collect(),
            b1: vec![0.0; hidden],
            w2: (0..hidden * output).map(|_| rng.uniform(b2_bound)).collect(),
            b2: vec![0.0; output],
        }
    }

    // Caller supplies scratch arrays to avoid allocating thousands of times.
    fn forward(&self, pixels: &[u8], activations: &mut [f32], probs: &mut [f32]) {
        debug_assert_eq!(pixels.len(), self.input);
        activations.copy_from_slice(&self.b1);
        for (i, &pixel) in pixels.iter().enumerate() {
            if pixel == 0 { continue; }
            let value = pixel as f32 / 255.0;
            for j in 0..self.hidden {
                activations[j] += value * self.w1[i * self.hidden + j];
            }
        }
        for a in activations.iter_mut() { *a = a.max(0.0); }
        probs.copy_from_slice(&self.b2);
        for (j, &value) in activations.iter().enumerate() {
            for k in 0..self.output {
                probs[k] += value * self.w2[j * self.output + k];
            }
        }
        let max_value = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for p in probs.iter_mut() {
            *p = (*p - max_value).exp();
            total += *p;
        }
        for p in probs.iter_mut() { *p /= total; }
    }

    // Backpropagation for one example, accumulate gradient for SGD minibatch.
    fn accumulate(&self, pixels: &[u8], label: usize, grads: &mut Gradients,
                  activations: &mut [f32], probs: &mut [f32], dhidden: &mut [f32]) -> f32 {
        self.forward(pixels, activations, probs);
        let loss = -probs[label].max(1e-12).ln();
        probs[label] -= 1.0; // gradient of softmax + cross-entropy
        for k in 0..self.output { grads.b2[k] += probs[k]; }
        for j in 0..self.hidden {
            let mut derivative = 0.0;
            for k in 0..self.output {
                grads.w2[j * self.output + k] += activations[j] * probs[k];
                derivative += self.w2[j * self.output + k] * probs[k];
            }
            dhidden[j] = if activations[j] > 0.0 { derivative } else { 0.0 };
            grads.b1[j] += dhidden[j];
        }
        for (i, &pixel) in pixels.iter().enumerate() {
            if pixel == 0 { continue; }
            let value = pixel as f32 / 255.0;
            for j in 0..self.hidden {
                grads.w1[i * self.hidden + j] += value * dhidden[j];
            }
        }
        loss
    }

    fn apply_sgd(&mut self, grads: &Gradients, scale: f32) {
        for (p, &g) in self.w1.iter_mut().zip(&grads.w1) { *p -= scale * g; }
        for (p, &g) in self.b1.iter_mut().zip(&grads.b1) { *p -= scale * g; }
        for (p, &g) in self.w2.iter_mut().zip(&grads.w2) { *p -= scale * g; }
        for (p, &g) in self.b2.iter_mut().zip(&grads.b2) { *p -= scale * g; }
    }
}

fn accuracy(network: &Network, ds: &Dataset, start: usize, end: usize) -> f32 {
    let mut activations = vec![0.0; network.hidden];
    let mut probs = vec![0.0; network.output];
    let mut correct = 0_usize;
    for idx in start..end {
        network.forward(ds.image(idx), &mut activations, &mut probs);
        let predicted = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        if predicted == ds.labels[idx] as usize { correct += 1; }
    }
    correct as f32 / (end - start) as f32
}

struct Config {
    data: PathBuf,
    samples: usize,
    val_size: usize,
    hidden: usize,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u32,
    output: PathBuf,
}
impl Config {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut result = Self {
            data: PathBuf::from("data"), samples: 10000, val_size: 1000,
            hidden: 64, epochs: 3, batch_size: 64, lr: 0.1, seed: 42,
            output: PathBuf::from("results/rust.json"),
        };
        let mut args = env::args().skip(1);
        while let Some(key) = args.next() {
            let value = args.next().ok_or(format!("Missing value for {key}"))?;
            match key.as_str() {
                "--data" => result.data = PathBuf::from(value),
                "--samples" => result.samples = value.parse()?,
                "--val-size" => result.val_size = value.parse()?,
                "--hidden" => result.hidden = value.parse()?,
                "--epochs" => result.epochs = value.parse()?,
                "--batch-size" => result.batch_size = value.parse()?,
                "--lr" => result.lr = value.parse()?,
                "--seed" => result.seed = value.parse()?,
                "--output" => result.output = PathBuf::from(value),
                _ => return Err(format!("Unrecognized flag {key}").into()),
            }
        }
        if result.samples == 0 || result.val_size == 0 || result.hidden == 0
            || result.epochs == 0 || result.batch_size == 0 || !result.lr.is_finite()
            || result.lr <= 0.0 {
            return Err("Samples, validation, hidden size, epochs, batch size and learning rate must be positive".into());
        }
        Ok(result)
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = Config::parse()?;
    let train = Dataset::read(&config.data, "train")?;
    let test = Dataset::read(&config.data, "t10k")?;
    if config.samples + config.val_size > train.count {
        return Err("Training and validation subsets overlap or exceed dataset".into());
    }
    let mut rng = XorShift32::new(config.seed);
    let mut network = Network::new(784, config.hidden, 10, &mut rng);
    let mut grads = Gradients::new(&network);
    let mut activations = vec![0.0; config.hidden];
    let mut probs = vec![0.0; 10];
    let mut dhidden = vec![0.0; config.hidden];
    let mut order: Vec<usize> = (0..config.samples).collect();
    let mut val_acc = 0.0;
    let started = Instant::now();
    for epoch in 1..=config.epochs {
        rng.shuffle(&mut order);
        let mut loss_total = 0.0;
        for batch in order.chunks(config.batch_size) {
            grads.clear();
            for &idx in batch {
                loss_total += network.accumulate(train.image(idx), train.labels[idx] as usize,
                    &mut grads, &mut activations, &mut probs, &mut dhidden);
            }
            network.apply_sgd(&grads, config.lr / batch.len() as f32);
        }
        val_acc = accuracy(&network, &train, train.count - config.val_size, train.count);
        println!("epoch {}/{} | loss={:.4} | val_acc={:.2}%",
                 epoch, config.epochs, loss_total / config.samples as f32, val_acc * 100.0);
    }
    let train_seconds = started.elapsed().as_secs_f64();
    let test_acc = accuracy(&network, &test, 0, test.count);
    let report = format!(concat!(
        "{{\n  \"language\": \"Rust\",\n  \"implementation\": \"Rust pure std (CPU)\",\n",
        "  \"samples\": {},\n  \"validation_samples\": {},\n  \"hidden\": {},\n",
        "  \"epochs\": {},\n  \"batch_size\": {},\n  \"learning_rate\": {},\n",
        "  \"seed\": {},\n  \"train_seconds\": {:.4},\n",
        "  \"val_accuracy\": {:.6},\n  \"test_accuracy\": {:.6}\n}}\n"
    ), config.samples, config.val_size, config.hidden, config.epochs, config.batch_size,
       config.lr, config.seed, train_seconds, val_acc, test_acc);
    if let Some(parent) = config.output.parent() { fs::create_dir_all(parent)?; }
    fs::write(&config.output, report)?;
    println!("test_acc={:.2}% | training={:.2}s | saved {}",
             test_acc * 100.0, train_seconds, config.output.display());
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_probability_distribution() {
        let mut rng = XorShift32::new(17);
        let net = Network::new(4, 5, 2, &mut rng);
        let mut hidden = vec![0.0; 5];
        let mut probs = vec![0.0; 2];
        net.forward(&[3, 255, 8, 90], &mut hidden, &mut probs);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|p| *p > 0.0 && *p < 1.0));
    }

    #[test]
    fn backprop_updates_weights_and_reduces_loss() {
        let mut rng = XorShift32::new(17);
        let mut net = Network::new(4, 8, 2, &mut rng);
        let examples: &[(&[u8], usize)] = &[
            (&[255, 0, 0, 0], 0), (&[220, 0, 10, 0], 0),
            (&[0, 255, 0, 0], 1), (&[0, 220, 0, 10], 1)
        ];
        let mut act = vec![0.0; 8];
        let mut prob = vec![0.0; 2];
        let mut dhidden = vec![0.0; 8];
        let loss = |net: &Network, act: &mut [f32], prob: &mut [f32]| -> f32 {
            examples.iter().map(|(image, target)| {
                net.forward(image, act, prob);
                -prob[*target].ln()
            }).sum::<f32>() / examples.len() as f32
        };
        let before = loss(&net, &mut act, &mut prob);
        let mut grads = Gradients::new(&net);
        for _ in 0..100 {
            grads.clear();
            for (pixels, target) in examples {
                net.accumulate(pixels, *target, &mut grads, &mut act, &mut prob, &mut dhidden);
            }
            net.apply_sgd(&grads, 0.2 / examples.len() as f32);
        }
        let after = loss(&net, &mut act, &mut prob);
        assert!(after < before, "loss failed to decrease: {before} -> {after}");
    }
}
