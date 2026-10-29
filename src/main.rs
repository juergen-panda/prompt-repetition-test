use log::{debug, error, info};
use rand::{RngExt, SeedableRng};
use schemars::JsonSchema;
use schemars::generate::SchemaSettings;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::time::Duration;

fn generate_json_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    SchemaSettings::openapi3()
        .with(|s| {
            s.meta_schema = None;
            s.inline_subschemas = true;
        })
        .into_generator()
        .into_root_schema_for::<T>()
        .to_value()
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct QueryResult<R> {
    #[schemars(description = "Result of the query")]
    result: R,
}

fn query_llm<T: schemars::JsonSchema + DeserializeOwned>(
    prompt: &str,
) -> Result<T, Box<dyn std::error::Error>> {
    let response_schema = generate_json_schema::<T>();

    let payload = serde_json::json!(
    {
        "seed": 0,
        "temperature": 0,
        "stream": false,
        // Switch off thinking in case of qwen models
        "chat_template_kwargs": {"enable_thinking": false},
        "thinking_budget_tokens": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name" : "ResponseSchema",
                "strict" : true,
                "schema" : response_schema,
            }
        },
    });

    debug!("Request data:\n{}", serde_json::to_string_pretty(&payload)?);

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;

    let result: serde_json::Value = client
        .post("http://localhost:9931/v1/chat/completions")
        .json(&payload)
        .send()?
        .json()?;

    debug!("Response:\n{result}");

    let content = result["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();

    let response: T = serde_json::from_str(&content)?;

    Ok(response)
}

#[derive(Debug)]
struct EvaluateMultiplePromptsResult {
    both_correct: u32,
    only_p_one_correct: u32,
    only_p_two_correct: u32,
    both_wrong: u32,
}

fn evaluate_prompts<
    T,
    R: Debug + schemars::JsonSchema + DeserializeOwned,
    PO: Fn(&T) -> String,
    PT: Fn(&T) -> String,
    E: Fn(&T, &R) -> bool,
>(
    data: &[T],
    prompt_one_fn: PO,
    prompt_two_fn: PT,
    evaluation_fn: E,
) -> EvaluateMultiplePromptsResult {
    let mut both_correct = 0;
    let mut p_one_correct = 0;
    let mut p_two_correct = 0;
    let mut both_wrong = 0;

    for data_point in data {
        let prompt = prompt_one_fn(data_point);

        let result_one_correct = match query_llm(&prompt) {
            Ok(result) => {
                let evaluation_result = evaluation_fn(data_point, &result);

                info!(
                    "Prompt: {prompt} with result: {result:?}. The result is correct? {evaluation_result}"
                );

                evaluation_result
            }
            Err(error) => {
                error!("Prompt: {prompt} results in error: {error:?}");
                false
            }
        };

        let prompt = prompt_two_fn(data_point);

        let result_two_correct = match query_llm(&prompt) {
            Ok(result) => {
                let evaluation_result = evaluation_fn(data_point, &result);

                info!(
                    "Prompt: {prompt} with result: {result:?}. The result is correct? {evaluation_result}"
                );

                evaluation_result
            }
            Err(error) => {
                error!("Prompt: {prompt} results in error: {error:?}");
                false
            }
        };

        match (result_one_correct, result_two_correct) {
            (true, true) => both_correct += 1,
            (true, false) => p_one_correct += 1,
            (false, true) => p_two_correct += 1,
            (false, false) => both_wrong += 1,
        }
    }

    EvaluateMultiplePromptsResult {
        both_correct,
        only_p_one_correct: p_one_correct,
        only_p_two_correct: p_two_correct,
        both_wrong,
    }
}

fn statistically_significant(p10: u64, p01: u64) -> bool {
    fn n_over_k(n: u64, k: u64) -> u64 {
        if k > n / 2 {
            return n_over_k(n, n - k);
        }

        let mut result = 1;

        for i in 1..=k {
            result *= n + 1 - i;
            result /= i;
        }

        result
    }

    let n = p10 + p01;

    if n == 0 {
        return false;
    }

    if n < 25 {
        return 2.
            * (0..=p01.min(p10))
                .map(|k| n_over_k(n, k) as f64 * 0.5_f64.powi(n as i32))
                .sum::<f64>()
            < 0.1;
    }

    // Compute statistical significance according to McNemar test using chi-squared distribution under significance level 10% and degree of freedom given by 1
    (((p10 as f64 - p01 as f64).abs() - 1.0).powi(2) / n as f64) > 2.706
}

#[test]
fn test_statistically_significant() {
    assert!(!statistically_significant(0, 0));

    assert!(statistically_significant(31, 17));
    assert!(statistically_significant(17, 31));
    assert!(!statistically_significant(30, 18));
    assert!(!statistically_significant(18, 30));
    assert!(!statistically_significant(24, 24));

    assert!(!statistically_significant(12, 12));
    assert!(statistically_significant(1, 9));
    assert!(statistically_significant(9, 1));
    assert!(!statistically_significant(2, 8));
    assert!(!statistically_significant(8, 2));
}

fn main() {
    env_logger::init();

    const TOTAL_ITERATIONS: usize = 500;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let data: Vec<_> = std::iter::repeat_with(|| {
        (
            rng.random_range(-1_000_000..=1_000_000),
            rng.random_range(-1_000_000..=1_000_000),
        )
    })
    .take(TOTAL_ITERATIONS)
    .collect();

    let result = evaluate_prompts(
        &data,
        |(a, b)| format!("What is the result of {a} + {b}?\n"),
        |(a, b)| format!("What is the result of {a} + {b}?\n").repeat(2),
        |(a, b), q: &QueryResult<i64>| q.result == a + b,
    );

    println!("=== Addition ===");
    println!(
        "Number of trials: {}, \
        Both approaches correct: {}, \
        Only single prompt correct: {}, \
        Only double prompt correct: {}, \
        All approaches failed: {}",
        data.len(),
        result.both_correct,
        result.only_p_one_correct,
        result.only_p_two_correct,
        result.both_wrong
    );
    println!(
        "Statistically significant: {}",
        statistically_significant(
            result.only_p_one_correct as u64,
            result.only_p_two_correct as u64
        )
    );

    let data: Vec<Vec<i64>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(|| rng.random_range(-1_000_000..=1_000_000))
            .take(10)
            .collect()
    })
    .take(TOTAL_ITERATIONS)
    .collect();

    let result = evaluate_prompts(
        &data,
        |a| format!("Given the following set {a:?}. What is the largest element in the set?\n"),
        |a| {
            format!("Given the following set {a:?}. What is the largest element in the set?\n")
                .repeat(2)
        },
        |a, q: &QueryResult<i64>| q.result == *a.iter().max().unwrap(),
    );

    println!("=== Largest Set Element ===");
    println!(
        "Number of trials: {}, \
        Both approaches correct: {}, \
        Only single prompt correct: {}, \
        Only double prompt correct: {}, \
        All approaches failed: {}",
        data.len(),
        result.both_correct,
        result.only_p_one_correct,
        result.only_p_two_correct,
        result.both_wrong
    );
    println!(
        "Statistically significant: {}",
        statistically_significant(
            result.only_p_one_correct as u64,
            result.only_p_two_correct as u64
        )
    );
}
