use std::sync::Arc;
use std::time::Duration;

use collection::common::adaptive_handle::AdaptiveSearchHandle;
use collection::config::{CollectionConfigInternal, CollectionParams, WalConfig};
use collection::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted,
};
use collection::operations::types::ShardStatus;
use collection::operations::universal_query::shard_query::{ScoringQuery, ShardQueryRequest};
use collection::operations::vector_params_builder::VectorParamsBuilder;
use collection::operations::{CollectionUpdateOperations, CreateIndex, FieldIndexOperations};
use collection::optimizers_builder::OptimizersConfig;
use collection::shards::local_shard::LocalShard;
use collection::shards::shard_trait::{ShardOperation, WaitUntil};
use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::save_on_disk::SaveOnDisk;
use criterion::{Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use segment::data_types::index::{TextIndexParams, TextIndexType};
use segment::data_types::vectors::{VectorStructInternal, only_default_vector};
use segment::fixtures::payload_fixtures::TEXT_KEY;
use segment::types::{
    Condition, Distance, FieldCondition, Filter, Payload, PayloadFieldSchema, PayloadSchemaParams,
    PayloadSchemaType, WithPayloadInterface, WithVector,
};
use serde_json::Map;
use shard::query::payload_query::{PayloadQueryInternal, TextQueryInternal};
use tempfile::{Builder, TempDir};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;
use tokio::time::{Instant, sleep};

#[cfg(not(target_os = "windows"))]
mod prof;

const DIM: usize = 16;
const DESIRED_SEGMENTS: usize = 8;
const POINTS_PER_SEGMENT: usize = 2_000;
const NUM_POINTS: usize = DESIRED_SEGMENTS * POINTS_PER_SEGMENT;
const TOKENS_PER_DOC: usize = 24;
const VOCAB_SIZE: usize = 8_192;
const QUERY_VARIANTS: usize = 64;
const TOP: usize = 10;
const TENANT_BUCKETS: usize = 32;
const UNIQUE_TERMS: [usize; 3] = [4, 16, 64];
const BATCH_SIZES: [usize; 3] = [1, 8, 32];
const SETUP_TIMEOUT: Duration = Duration::from_secs(90);
const BENCH_TIMEOUT: Duration = Duration::from_secs(30);

fn vocabulary() -> Vec<String> {
    (0..VOCAB_SIZE).map(|idx| format!("tok{idx:05}")).collect()
}

fn point_vector(point_id: usize) -> Vec<f32> {
    (0..DIM)
        .map(|dim| ((point_id.wrapping_mul(31).wrapping_add(dim * 17)) % 100) as f32 / 100.0)
        .collect()
}

fn make_document(vocab: &[String], point_id: usize) -> String {
    let start = point_id.wrapping_mul(13) % vocab.len();
    (0..TOKENS_PER_DOC)
        .map(|offset| vocab[(start + offset * 29) % vocab.len()].as_str())
        .join(" ")
}

fn make_query_strings(vocab: &[String], unique_terms: usize) -> Vec<String> {
    (0..QUERY_VARIANTS)
        .map(|seed| {
            (0..unique_terms)
                .map(|offset| vocab[(seed * 17 + offset * 31) % vocab.len()].as_str())
                .join(" ")
        })
        .collect()
}

fn create_text_index_operation() -> CollectionUpdateOperations {
    let field_schema =
        PayloadFieldSchema::FieldParams(PayloadSchemaParams::Text(TextIndexParams {
            r#type: TextIndexType::Text,
            enable_score: Some(true),
            ..Default::default()
        }));

    CollectionUpdateOperations::FieldIndexOperation(FieldIndexOperations::CreateIndex(
        CreateIndex {
            field_name: TEXT_KEY.parse().unwrap(),
            field_schema: Some(field_schema),
        },
    ))
}

fn create_tenant_index_operation() -> CollectionUpdateOperations {
    CollectionUpdateOperations::FieldIndexOperation(FieldIndexOperations::CreateIndex(
        CreateIndex {
            field_name: "tenant".parse().unwrap(),
            field_schema: Some(PayloadFieldSchema::FieldType(PayloadSchemaType::Keyword)),
        },
    ))
}

async fn wait_optimization(shard: &LocalShard, timeout: Duration) {
    let start = Instant::now();

    loop {
        let (status, _) = shard.local_shard_status().await;
        let has_proxy = shard
            .segments()
            .read()
            .iter()
            .any(|(_, segment)| !segment.is_original());

        if status == ShardStatus::Green && !has_proxy {
            return;
        }

        assert!(
            start.elapsed() < timeout,
            "payload query benchmark setup timed out while waiting for optimization"
        );
        sleep(Duration::from_millis(100)).await;
    }
}

fn setup() -> (
    TempDir,
    LocalShard,
    Runtime,
    usize,
    Vec<(usize, Vec<String>)>,
    Filter,
) {
    let storage_dir = Builder::new()
        .prefix("payload-query-bench")
        .tempdir()
        .unwrap();

    let runtime = Runtime::new().unwrap();
    let handle = runtime.handle().clone();

    let wal_config = WalConfig {
        wal_capacity_mb: 1,
        wal_segments_ahead: 0,
        wal_retain_closed: 1,
    };

    let collection_params = CollectionParams {
        vectors: VectorParamsBuilder::new(DIM as u64, Distance::Dot)
            .build()
            .into(),
        ..CollectionParams::empty()
    };

    let collection_config = CollectionConfigInternal {
        params: collection_params,
        optimizer_config: OptimizersConfig {
            deleted_threshold: 0.9,
            vacuum_min_vector_number: 1_000,
            default_segment_number: DESIRED_SEGMENTS,
            max_segment_size: Some(POINTS_PER_SEGMENT),
            #[expect(deprecated)]
            memmap_threshold: Some(POINTS_PER_SEGMENT),
            indexing_threshold: Some((POINTS_PER_SEGMENT / 2).max(1)),
            flush_interval_sec: 0,
            max_optimization_threads: Some(4),
            prevent_unoptimized: None,
        },
        wal_config,
        hnsw_config: Default::default(),
        quantization_config: Default::default(),
        strict_mode_config: Default::default(),
        uuid: None,
        metadata: None,
    };

    let optimizers_config = collection_config.optimizer_config.clone();
    let shared_config = Arc::new(RwLock::new(collection_config));

    let payload_index_schema_dir = Builder::new()
        .prefix("payload-query-schema")
        .tempdir()
        .unwrap();
    let payload_index_schema_file = payload_index_schema_dir.path().join("payload-schema.json");
    let payload_index_schema =
        Arc::new(SaveOnDisk::load_or_init_default(payload_index_schema_file).unwrap());

    let shard = handle
        .block_on(LocalShard::build_local(
            0,
            "payload_query_bench".to_string(),
            storage_dir.path(),
            shared_config,
            Default::default(),
            payload_index_schema,
            handle.clone(),
            AdaptiveSearchHandle::new_fixed(handle.clone()),
            ResourceBudget::default(),
            optimizers_config,
        ))
        .unwrap();

    handle
        .block_on(shard.update(
            create_text_index_operation().into(),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
        ))
        .unwrap();
    handle
        .block_on(shard.update(
            create_tenant_index_operation().into(),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
        ))
        .unwrap();

    let vocab = vocabulary();

    for chunk_idx in 0..DESIRED_SEGMENTS {
        let start = chunk_idx * POINTS_PER_SEGMENT;
        let end = start + POINTS_PER_SEGMENT;
        let points = (start..end)
            .map(|point_id| {
                let mut payload_map = Map::new();
                payload_map.insert(TEXT_KEY.to_string(), make_document(&vocab, point_id).into());
                payload_map.insert(
                    "tenant".to_string(),
                    format!("tenant-{}", point_id % TENANT_BUCKETS).into(),
                );

                let vector = point_vector(point_id);
                let vectors = only_default_vector(&vector);

                PointStructPersisted {
                    id: (point_id as u64).into(),
                    vector: VectorStructInternal::from(vectors).into(),
                    payload: Some(Payload(payload_map)),
                }
            })
            .collect();

        handle
            .block_on(
                shard.update(
                    CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
                        PointInsertOperationsInternal::PointsList(points),
                    ))
                    .into(),
                    WaitUntil::Visible,
                    None,
                    HwMeasurementAcc::new(),
                ),
            )
            .unwrap();

        shard.trigger_optimizers();
        handle.block_on(wait_optimization(&shard, SETUP_TIMEOUT));
    }

    let segment_count = shard.segments().read().len();
    eprintln!(
        "payload_query_bench setup: num_points={NUM_POINTS}, desired_segments={DESIRED_SEGMENTS}, actual_segments={segment_count}"
    );

    let query_sets = UNIQUE_TERMS
        .into_iter()
        .map(|unique_terms| (unique_terms, make_query_strings(&vocab, unique_terms)))
        .collect();

    let filter = Filter::new_must(Condition::Field(FieldCondition::new_match(
        "tenant".parse().unwrap(),
        "tenant-0".to_owned().into(),
    )));

    (
        storage_dir,
        shard,
        runtime,
        segment_count,
        query_sets,
        filter,
    )
}

fn payload_query_bench(c: &mut Criterion) {
    let (_tempdir, shard, runtime, segment_count, query_sets, filter) = setup();
    let search_runtime_handle = AdaptiveSearchHandle::new_fixed(runtime.handle().clone());
    let text_key: segment::json_path::JsonPath = TEXT_KEY.parse().unwrap();

    let scenarios: Vec<(&str, Option<Filter>)> =
        vec![("unfiltered", None), ("filtered", Some(filter.clone()))];

    let mut single_group =
        c.benchmark_group(format!("payload_query/single/segments-{segment_count}"));
    single_group.sample_size(10);

    for (scenario_name, scenario_filter) in &scenarios {
        for (unique_terms, queries) in &query_sets {
            let mut query_idx = 0usize;
            let text_key = text_key.clone();
            let scenario_filter = scenario_filter.clone();
            single_group.bench_function(
                format!("{scenario_name}/unique_terms-{unique_terms}"),
                |b| {
                    b.iter(|| {
                        let query_str = queries[query_idx % queries.len()].clone();
                        query_idx += 1;

                        runtime.block_on(async {
                            let results = shard
                                .search_with_payload_query(
                                    PayloadQueryInternal::Text(TextQueryInternal {
                                        key: text_key.clone(),
                                        query_str,
                                    }),
                                    scenario_filter.clone(),
                                    TOP,
                                    None,
                                    BENCH_TIMEOUT,
                                    HwMeasurementAcc::new(),
                                )
                                .await
                                .unwrap();
                            assert!(results.len() <= TOP);
                        });
                    })
                },
            );
        }
    }

    single_group.finish();

    let mut batch_group =
        c.benchmark_group(format!("payload_query/batch/segments-{segment_count}"));
    batch_group.sample_size(10);

    for (scenario_name, scenario_filter) in &scenarios {
        for (unique_terms, queries) in &query_sets {
            for &batch_size in &BATCH_SIZES {
                let request_sets = (0..QUERY_VARIANTS.min(8))
                    .map(|seed| {
                        Arc::new(
                            (0..batch_size)
                                .map(|idx| ShardQueryRequest {
                                    prefetches: vec![],
                                    query: Some(ScoringQuery::Payload(PayloadQueryInternal::Text(
                                        TextQueryInternal {
                                            key: text_key.clone(),
                                            query_str: queries[(seed + idx) % queries.len()]
                                                .clone(),
                                        },
                                    ))),
                                    filter: scenario_filter.clone(),
                                    params: None,
                                    limit: TOP,
                                    offset: 0,
                                    with_payload: WithPayloadInterface::Bool(false),
                                    with_vector: WithVector::Bool(false),
                                    score_threshold: None,
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut request_set_idx = 0usize;

                batch_group.bench_function(
                    format!("{scenario_name}/unique_terms-{unique_terms}/batch-{batch_size}"),
                    |b| {
                        b.iter(|| {
                            let requests =
                                request_sets[request_set_idx % request_sets.len()].clone();
                            request_set_idx += 1;
                            runtime.block_on(async {
                                let results = shard
                                    .query_batch(
                                        requests,
                                        &search_runtime_handle,
                                        Some(BENCH_TIMEOUT),
                                        HwMeasurementAcc::new(),
                                    )
                                    .await
                                    .unwrap();
                                assert_eq!(results.len(), batch_size);
                            });
                        })
                    },
                );
            }
        }
    }

    batch_group.finish();

    runtime.block_on(async {
        shard.stop_gracefully().await;
    });
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(prof::FlamegraphProfiler::new(100));
    targets = payload_query_bench
}

#[cfg(target_os = "windows")]
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = payload_query_bench,
}

criterion_main!(benches);
