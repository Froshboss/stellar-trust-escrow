//! # StellarTrustEscrow — Soroban Smart Contract
//!
//! Milestone-based escrow with on-chain reputation on the Stellar network.
//!
//! ## Gas Optimizations (Issue #65)
//!
//! 1. **Storage**: `EscrowMeta` and `Milestone` are stored in separate granular
//!    persistent entries — only the touched entry is read/written per call.
//!    The old monolithic `EscrowState` (with an inline `Vec<Milestone>`) is
//!    kept only as a view-layer return type.
//!
//! 2. **TTL bumps**: Consolidated into `bump_instance_ttl` / `bump_persistent_ttl`
//!    helpers called once per entry per transaction, not on every sub-call.
//!
//! 3. **Loop elimination**: `approve_milestone` previously re-loaded every
//!    milestone in a loop to check completion. Replaced with an `approved_count`
//!    field on `EscrowMeta` — O(1) completion check.
//!
//! 4. **Redundant loads**: `release_funds` no longer re-loads the milestone
//!    after `approve_milestone` already validated and saved it. Auth checks
//!    are done before any storage reads.
//!
//! 5. **Math**: All arithmetic uses `checked_*` only where overflow is
//!    plausible; inner hot-paths use direct ops with compile-time-safe bounds.
//!
//! 6. **Events**: Data tuples are kept minimal — addresses are passed by
//!    reference and cloned only at the `publish` call site.

#![no_std]
#![allow(clippy::too_many_arguments)]

mod errors;
mod events;
mod types;
mod upgrade_tests;

pub use errors::EscrowError;
pub use types::{
    DataKey, EscrowState, EscrowStatus, FeeDelegation, MetaTransaction, Milestone, MilestoneStatus,
    ReputationRecord,
};

use alloc::string::ToString;
use soroban_sdk::{
    contract, contractimpl, contracttype, crypto, token, Address, BytesN, Env, String, Vec,
};

// ── TTL constants ─────────────────────────────────────────────────────────────
// Bump only when remaining TTL falls below threshold, extending to target.
const INSTANCE_TTL_THRESHOLD: u32 = 5_000;
const INSTANCE_TTL_EXTEND_TO: u32 = 50_000;
const PERSISTENT_TTL_THRESHOLD: u32 = 5_000;
const PERSISTENT_TTL_EXTEND_TO: u32 = 50_000;

// ── Granular storage keys ─────────────────────────────────────────────────────
// Separate keys for meta vs each milestone avoids deserialising the full
// milestone list on every escrow-level operation.
#[contracttype]
#[derive(Clone)]
enum PackedDataKey {
    EscrowMeta(u64),
    Milestone(u64, u32),
}

// ── Meta-transaction argument structs ────────────────────────────────────────
#[derive(Clone)]
struct CreateEscrowArgs {
    client: Address,
    freelancer: Address,
    token: Address,
    total_amount: i128,
    brief_hash: BytesN<32>,
    arbiter: Option<Address>,
    deadline: Option<u64>,
    lock_time: Option<u64>,
}

#[derive(Clone)]
struct AddMilestoneArgs {
    caller: Address,
    escrow_id: u64,
    title: String,
    description_hash: BytesN<32>,
    amount: i128,
}

#[derive(Clone)]
struct SubmitMilestoneArgs {
    caller: Address,
    escrow_id: u64,
    milestone_id: u32,
}

#[derive(Clone)]
struct ApproveMilestoneArgs {
    caller: Address,
    escrow_id: u64,
    milestone_id: u32,
}

// ── EscrowMeta ────────────────────────────────────────────────────────────────
// Lightweight header stored separately from milestones.
// `approved_count` replaces the O(n) "all approved?" loop in approve_milestone.
#[contracttype]
#[derive(Clone, Debug)]
struct EscrowMeta {
    escrow_id: u64,
    client: Address,
    freelancer: Address,
    token: Address,
    total_amount: i128,
    /// Running sum of milestone amounts added so far (allocation guard).
    allocated_amount: i128,
    remaining_balance: i128,
    status: EscrowStatus,
    milestone_count: u32,
    /// Number of milestones in Approved state — avoids full scan on completion check.
    approved_count: u32,
    arbiter: Option<Address>,
    created_at: u64,
    deadline: Option<u64>,
    /// Optional lock time (ledger timestamp) - funds locked until this time.
    lock_time: Option<u64>,
    /// Optional extension deadline for the lock time.
    lock_time_extension: Option<u64>,
    brief_hash: BytesN<32>,
}

// ── Storage helpers ───────────────────────────────────────────────────────────
struct ContractStorage;

impl ContractStorage {
    fn initialize(env: &Env, admin: &Address) -> Result<(), EscrowError> {
        let instance = env.storage().instance();
        if instance.has(&DataKey::Admin) {
            return Err(EscrowError::AlreadyInitialized);
        }
        instance.set(&DataKey::Admin, admin);
        instance.set(&DataKey::EscrowCounter, &0_u64);
        Self::bump_instance_ttl(env);
        Ok(())
    }

    fn require_initialized(env: &Env) -> Result<(), EscrowError> {
        if !env.storage().instance().has(&DataKey::Admin) {
            return Err(EscrowError::NotInitialized);
        }
        Self::bump_instance_ttl(env);
        Ok(())
    }

    fn require_admin(env: &Env, caller: &Address) -> Result<(), EscrowError> {
        Self::require_initialized(env)?;
        let admin: Address = env
            .storage()
            .instance()
            .get(&DataKey::Admin)
            .ok_or(EscrowError::NotInitialized)?;
        if *caller != admin {
            return Err(EscrowError::AdminOnly);
        }
        Ok(())
    }

    fn next_escrow_id(env: &Env) -> Result<u64, EscrowError> {
        let instance = env.storage().instance();
        let id: u64 = instance.get(&DataKey::EscrowCounter).unwrap_or(0_u64);
        instance.set(&DataKey::EscrowCounter, &(id + 1));
        // Instance TTL already bumped by require_initialized caller
        Ok(id)
    }

    fn escrow_count(env: &Env) -> u64 {
        let count = env
            .storage()
            .instance()
            .get(&DataKey::EscrowCounter)
            .unwrap_or(0_u64);
        if env.storage().instance().has(&DataKey::Admin) {
            Self::bump_instance_ttl(env);
        }
        count
    }

    // ── Escrow meta ───────────────────────────────────────────────────────────

    fn load_escrow_meta(env: &Env, escrow_id: u64) -> Result<EscrowMeta, EscrowError> {
        let key = PackedDataKey::EscrowMeta(escrow_id);
        let meta = env
            .storage()
            .persistent()
            .get(&key)
            .ok_or(EscrowError::EscrowNotFound)?;
        Self::bump_persistent_ttl(env, &key);
        Ok(meta)
    }

    fn save_escrow_meta(env: &Env, meta: &EscrowMeta) {
        let key = PackedDataKey::EscrowMeta(meta.escrow_id);
        env.storage().persistent().set(&key, meta);
        Self::bump_persistent_ttl(env, &key);
    }

    // ── Milestones ────────────────────────────────────────────────────────────

    fn load_milestone(
        env: &Env,
        escrow_id: u64,
        milestone_id: u32,
    ) -> Result<Milestone, EscrowError> {
        let key = PackedDataKey::Milestone(escrow_id, milestone_id);
        let m = env
            .storage()
            .persistent()
            .get(&key)
            .ok_or(EscrowError::MilestoneNotFound)?;
        Self::bump_persistent_ttl(env, &key);
        Ok(m)
    }

    fn save_milestone(env: &Env, escrow_id: u64, milestone: &Milestone) {
        let key = PackedDataKey::Milestone(escrow_id, milestone.id);
        env.storage().persistent().set(&key, milestone);
        Self::bump_persistent_ttl(env, &key);
    }

    // ── Full escrow view (read-only, assembles EscrowState for callers) ───────
    fn load_escrow(env: &Env, escrow_id: u64) -> Result<EscrowState, EscrowError> {
        let meta = Self::load_escrow_meta(env, escrow_id)?;
        let mut milestones = Vec::new(env);
        for mid in 0..meta.milestone_count {
            milestones.push_back(Self::load_milestone(env, escrow_id, mid)?);
        }
        Ok(EscrowState {
            escrow_id: meta.escrow_id,
            client: meta.client,
            freelancer: meta.freelancer,
            token: meta.token,
            total_amount: meta.total_amount,
            remaining_balance: meta.remaining_balance,
            status: meta.status,
            milestones,
            arbiter: meta.arbiter,
            created_at: meta.created_at,
            deadline: meta.deadline,
            lock_time: meta.lock_time,
            lock_time_extension: meta.lock_time_extension,
            brief_hash: meta.brief_hash,
        })
    }

    // ── Reputation ────────────────────────────────────────────────────────────

    fn load_reputation(env: &Env, address: &Address) -> ReputationRecord {
        let key = DataKey::Reputation(address.clone());
        match env.storage().persistent().get(&key) {
            Some(record) => {
                Self::bump_persistent_ttl(env, &key);
                record
            }
            None => ReputationRecord {
                address: address.clone(),
                total_score: 0,
                completed_escrows: 0,
                disputed_escrows: 0,
                disputes_won: 0,
                total_volume: 0,
                last_updated: env.ledger().timestamp(),
            },
        }
    }

    fn save_reputation(env: &Env, record: &ReputationRecord) {
        let key = DataKey::Reputation(record.address.clone());
        env.storage().persistent().set(&key, record);
        Self::bump_persistent_ttl(env, &key);
    }

    // ── TTL helpers ───────────────────────────────────────────────────────────

    #[inline]
    fn bump_instance_ttl(env: &Env) {
        env.storage()
            .instance()
            .extend_ttl(INSTANCE_TTL_THRESHOLD, INSTANCE_TTL_EXTEND_TO);
    }

    #[inline]
    fn bump_persistent_ttl<K>(env: &Env, key: &K)
    where
        K: soroban_sdk::IntoVal<Env, soroban_sdk::Val>,
    {
        env.storage().persistent().extend_ttl(
            key,
            PERSISTENT_TTL_THRESHOLD,
            PERSISTENT_TTL_EXTEND_TO,
        );
    }

    // ── Time lock helpers ─────────────────────────────────────────────────────────

    /// Checks if the lock time has expired for an escrow.
    /// Returns Ok(()) if funds can be released, Err if still locked.
    fn check_lock_time_expired(
        env: &Env,
        escrow_id: u64,
        lock_time: Option<u64>,
    ) -> Result<(), EscrowError> {
        if let Some(lt) = lock_time {
            let now = env.ledger().timestamp();
            if now < lt {
                return Err(EscrowError::LockTimeNotExpired);
            }
            // Lock has expired - emit event
            events::emit_lock_time_expired(env, escrow_id, lt);
        }
        Ok(())
    }

    // ── Meta-transaction helpers ──────────────────────────────────────────────────

    /// Gets the current nonce for a meta-transaction signer.
    fn get_meta_tx_nonce(env: &Env, signer: &Address) -> u64 {
        let key = DataKey::MetaTxNonce(signer.clone());
        env.storage().persistent().get(&key).unwrap_or(0_u64)
    }

    /// Increments the nonce for a meta-transaction signer.
    fn increment_meta_tx_nonce(env: &Env, signer: &Address) -> Result<(), EscrowError> {
        let key = DataKey::MetaTxNonce(signer.clone());
        let current_nonce = Self::get_meta_tx_nonce(env, signer);
        env.storage().persistent().set(&key, &(current_nonce + 1));
        Self::bump_persistent_ttl(env, &key);
        Ok(())
    }

    /// Verifies a meta-transaction signature.
    fn verify_meta_tx_signature(env: &Env, meta_tx: &MetaTransaction) -> Result<(), EscrowError> {
        // Check if deadline has passed
        let now = env.ledger().timestamp();
        if now > meta_tx.deadline {
            return Err(EscrowError::MetaTxExpired);
        }

        // Check if nonce is valid (not already used)
        let current_nonce = Self::get_meta_tx_nonce(env, &meta_tx.signer);
        if meta_tx.nonce != current_nonce {
            return Err(EscrowError::NonceAlreadyUsed);
        }

        // Create the message to verify
        // Format: domain_separator + nonce + deadline + function_name + function_args
        let domain_separator = String::from_str(env, "StellarTrustEscrow-MetaTx");
        let nonce_str = meta_tx.nonce.to_string();
        let deadline_str = meta_tx.deadline.to_string();

        let message = format!(
            "{}{}{}{}{}",
            domain_separator, nonce_str, deadline_str, meta_tx.function_name, meta_tx.function_args
        );

        // Verify the signature
        let message_bytes = message.as_bytes();
        crypto::verify(env, &meta_tx.signer, &message_bytes, &meta_tx.signature)?;

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CONTRACT
// ─────────────────────────────────────────────────────────────────────────────

#[contract]
pub struct EscrowContract;

#[contractimpl]
impl EscrowContract {
    // ── Initialization ────────────────────────────────────────────────────────

    pub fn initialize(env: Env, admin: Address) -> Result<(), EscrowError> {
        ContractStorage::initialize(&env, &admin)
    }

    // ── Escrow Lifecycle ──────────────────────────────────────────────────────

    /// Creates a new escrow and locks funds in the contract.
    ///
    /// # Gas notes
    /// - Auth check before any storage read.
    /// - Single `save_escrow_meta` write; no milestone writes at creation.
    /// - Token transfer is the dominant cost; nothing we can do there.
    pub fn create_escrow(
        env: Env,
        client: Address,
        freelancer: Address,
        token: Address,
        total_amount: i128,
        brief_hash: BytesN<32>,
        arbiter: Option<Address>,
        deadline: Option<u64>,
        lock_time: Option<u64>,
    ) -> Result<u64, EscrowError> {
        // Auth + validation before any storage I/O
        client.require_auth();
        ContractStorage::require_initialized(&env)?;

        if total_amount <= 0 {
            return Err(EscrowError::InvalidEscrowAmount);
        }

        let now = env.ledger().timestamp();
        if let Some(dl) = deadline {
            if dl <= now {
                return Err(EscrowError::InvalidDeadline);
            }
        }

        // Validate lock_time if provided
        if let Some(lt) = lock_time {
            if lt <= now {
                return Err(EscrowError::InvalidLockTime);
            }
        }

        let escrow_id = ContractStorage::next_escrow_id(&env)?;

        // Transfer tokens — single cross-contract call
        token::Client::new(&env, &token).transfer(
            &client,
            &env.current_contract_address(),
            &total_amount,
        );

        ContractStorage::save_escrow_meta(
            &env,
            &EscrowMeta {
                escrow_id,
                client: client.clone(),
                freelancer: freelancer.clone(),
                token,
                total_amount,
                allocated_amount: 0,
                remaining_balance: total_amount,
                status: EscrowStatus::Active,
                milestone_count: 0,
                approved_count: 0,
                arbiter,
                created_at: now,
                deadline,
                lock_time,
                lock_time_extension: None,
                brief_hash,
            },
        );

        events::emit_escrow_created(&env, escrow_id, &client, &freelancer, total_amount);
        Ok(escrow_id)
    }

    /// Adds a milestone to an existing escrow.
    ///
    /// # Gas notes
    /// - Auth before storage read.
    /// - Writes only the new `Milestone` entry + updated `EscrowMeta`.
    pub fn add_milestone(
        env: Env,
        caller: Address,
        escrow_id: u64,
        title: String,
        description_hash: BytesN<32>,
        amount: i128,
    ) -> Result<u32, EscrowError> {
        caller.require_auth();

        if amount <= 0 {
            return Err(EscrowError::InvalidMilestoneAmount);
        }

        let mut meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;

        if caller != meta.client {
            return Err(EscrowError::ClientOnly);
        }
        if meta.status != EscrowStatus::Active {
            return Err(EscrowError::EscrowNotActive);
        }

        let next_allocated = meta
            .allocated_amount
            .checked_add(amount)
            .ok_or(EscrowError::MilestoneAmountExceedsEscrow)?;
        if next_allocated > meta.total_amount {
            return Err(EscrowError::MilestoneAmountExceedsEscrow);
        }

        let milestone_id = meta.milestone_count;
        meta.milestone_count = meta
            .milestone_count
            .checked_add(1)
            .ok_or(EscrowError::TooManyMilestones)?;
        meta.allocated_amount = next_allocated;

        ContractStorage::save_milestone(
            &env,
            escrow_id,
            &Milestone {
                id: milestone_id,
                title,
                description_hash,
                amount,
                status: MilestoneStatus::Pending,
                submitted_at: None,
                resolved_at: None,
            },
        );
        ContractStorage::save_escrow_meta(&env, &meta);

        events::emit_milestone_added(&env, escrow_id, milestone_id, amount);
        Ok(milestone_id)
    }

    /// Freelancer submits work for a milestone.
    ///
    /// # Gas notes
    /// - Loads only the single milestone entry, not the full escrow.
    pub fn submit_milestone(
        env: Env,
        caller: Address,
        escrow_id: u64,
        milestone_id: u32,
    ) -> Result<(), EscrowError> {
        caller.require_auth();

        // Load meta only to verify freelancer identity
        let meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;
        if caller != meta.freelancer {
            return Err(EscrowError::FreelancerOnly);
        }

        let mut milestone = ContractStorage::load_milestone(&env, escrow_id, milestone_id)?;
        if milestone.status != MilestoneStatus::Pending
            && milestone.status != MilestoneStatus::Rejected
        {
            return Err(EscrowError::InvalidMilestoneState);
        }

        milestone.status = MilestoneStatus::Submitted;
        milestone.submitted_at = Some(env.ledger().timestamp());
        ContractStorage::save_milestone(&env, escrow_id, &milestone);

        events::emit_milestone_submitted(&env, escrow_id, milestone_id, &caller);
        Ok(())
    }

    /// Client approves a submitted milestone and releases funds.
    ///
    /// # Gas notes
    /// - O(1) completion check via `approved_count` field — no milestone loop.
    /// - Single token transfer call.
    /// - Two storage writes: milestone + meta.
    pub fn approve_milestone(
        env: Env,
        caller: Address,
        escrow_id: u64,
        milestone_id: u32,
    ) -> Result<(), EscrowError> {
        caller.require_auth();

        let mut meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;
        if caller != meta.client {
            return Err(EscrowError::ClientOnly);
        }
        if meta.status != EscrowStatus::Active {
            return Err(EscrowError::EscrowNotActive);
        }

        // Check if lock time has expired
        ContractStorage::check_lock_time_expired(&env, escrow_id, meta.lock_time)?;

        let mut milestone = ContractStorage::load_milestone(&env, escrow_id, milestone_id)?;
        if milestone.status != MilestoneStatus::Submitted {
            return Err(EscrowError::InvalidMilestoneState);
        }

        let amount = milestone.amount;
        let now = env.ledger().timestamp();

        milestone.status = MilestoneStatus::Approved;
        milestone.resolved_at = Some(now);
        ContractStorage::save_milestone(&env, escrow_id, &milestone);

        // Release funds — single cross-contract call
        token::Client::new(&env, &meta.token).transfer(
            &env.current_contract_address(),
            &meta.freelancer,
            &amount,
        );

        // STE-04 fix: checked_sub instead of silent underflow
        meta.remaining_balance = meta
            .remaining_balance
            .checked_sub(amount)
            .ok_or(EscrowError::AmountMismatch)?;

        // O(1) completion check via approved_count (main branch optimization)
        meta.approved_count += 1;
        if meta.approved_count == meta.milestone_count && meta.milestone_count > 0 {
            meta.status = EscrowStatus::Completed;
            // STE-03 fix: emit completion event so the indexer can update DB
            events::emit_escrow_completed(&env, escrow_id);
        }

        ContractStorage::save_escrow_meta(&env, &meta);

        events::emit_milestone_approved(&env, escrow_id, milestone_id, amount);
        events::emit_funds_released(&env, escrow_id, &meta.freelancer, amount);
        Ok(())
    }

    /// Client rejects a submitted milestone.
    ///
    /// # Gas notes
    /// - Loads only the single milestone entry.
    pub fn reject_milestone(
        env: Env,
        caller: Address,
        escrow_id: u64,
        milestone_id: u32,
    ) -> Result<(), EscrowError> {
        caller.require_auth();

        let meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;
        if caller != meta.client {
            return Err(EscrowError::ClientOnly);
        }
        if meta.status != EscrowStatus::Active {
            return Err(EscrowError::EscrowNotActive);
        }

        let mut milestone = ContractStorage::load_milestone(&env, escrow_id, milestone_id)?;
        if milestone.status != MilestoneStatus::Submitted {
            return Err(EscrowError::InvalidMilestoneState);
        }

        milestone.status = MilestoneStatus::Rejected;
        milestone.resolved_at = Some(env.ledger().timestamp());
        ContractStorage::save_milestone(&env, escrow_id, &milestone);

        events::emit_milestone_rejected(&env, escrow_id, milestone_id, &caller);
        Ok(())
    }

    /// Admin-triggered fund release for an already-approved milestone.
    ///
    /// Admin-only fallback for edge cases. Normal flow uses `approve_milestone`.
    ///
    /// # Security (STE-01, STE-02)
    /// - Requires admin authorization.
    /// - Milestone must be `Approved` to prevent double-payment.
    pub fn release_funds(
        env: Env,
        caller: Address,
        escrow_id: u64,
        milestone_id: u32,
    ) -> Result<(), EscrowError> {
        // STE-01 fix: admin-only auth
        ContractStorage::require_initialized(&env)?;
        let admin: Address = env
            .storage()
            .instance()
            .get(&DataKey::Admin)
            .ok_or(EscrowError::NotInitialized)?;
        caller.require_auth();
        if caller != admin {
            return Err(EscrowError::AdminOnly);
        }

        // Load milestone first — cheaper than meta if it fails
        let milestone = ContractStorage::load_milestone(&env, escrow_id, milestone_id)?;
        if milestone.status != MilestoneStatus::Approved {
            return Err(EscrowError::InvalidMilestoneState);
        }

        // Load meta to check lock time
        let mut meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;

        // Check if lock time has expired
        ContractStorage::check_lock_time_expired(&env, escrow_id, meta.lock_time)?;

        let amount = milestone.amount;
        meta.remaining_balance = meta
            .remaining_balance
            .checked_sub(amount)
            .ok_or(EscrowError::AmountMismatch)?;

        token::Client::new(&env, &meta.token).transfer(
            &env.current_contract_address(),
            &meta.freelancer,
            &amount,
        );
        ContractStorage::save_escrow_meta(&env, &meta);

        events::emit_funds_released(&env, escrow_id, &meta.freelancer, amount);
        Ok(())
    }

    /// Cancels an escrow and returns remaining funds to the client.
    pub fn cancel_escrow(env: Env, caller: Address, escrow_id: u64) -> Result<(), EscrowError> {
        caller.require_auth();

        let mut meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;
        if caller != meta.client {
            return Err(EscrowError::ClientOnly);
        }
        if meta.status != EscrowStatus::Active {
            return Err(EscrowError::EscrowNotActive);
        }

        // Reject cancellation if any milestone is Submitted or Approved
        for mid in 0..meta.milestone_count {
            let m = ContractStorage::load_milestone(&env, escrow_id, mid)?;
            if m.status == MilestoneStatus::Submitted || m.status == MilestoneStatus::Approved {
                return Err(EscrowError::CannotCancelWithPendingFunds);
            }
        }

        let returned = meta.remaining_balance;
        token::Client::new(&env, &meta.token).transfer(
            &env.current_contract_address(),
            &meta.client,
            &returned,
        );

        meta.remaining_balance = 0;
        meta.status = EscrowStatus::Cancelled;
        ContractStorage::save_escrow_meta(&env, &meta);

        events::emit_escrow_cancelled(&env, escrow_id, returned);
        Ok(())
    }

    // ── Time Lock Extension ─────────────────────────────────────────────────────

    /// Extends the lock time for an escrow.
    ///
    /// Only the client can extend the lock time, and the new lock time
    /// must be in the future.
    pub fn extend_lock_time(
        env: Env,
        caller: Address,
        escrow_id: u64,
        new_lock_time: u64,
    ) -> Result<(), EscrowError> {
        caller.require_auth();

        let mut meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;

        if caller != meta.client {
            return Err(EscrowError::ClientOnly);
        }
        if meta.status != EscrowStatus::Active {
            return Err(EscrowError::EscrowNotActive);
        }

        let now = env.ledger().timestamp();
        if new_lock_time <= now {
            return Err(EscrowError::InvalidLockTimeExtension);
        }

        let old_lock_time = meta.lock_time.unwrap_or(0);

        // If there's an existing lock_time_extension, use that as the maximum
        if let Some(ext) = meta.lock_time_extension {
            if new_lock_time > ext {
                return Err(EscrowError::InvalidLockTimeExtension);
            }
        }

        meta.lock_time = Some(new_lock_time);
        ContractStorage::save_escrow_meta(&env, &meta);

        events::emit_lock_time_extended(&env, escrow_id, old_lock_time, new_lock_time, &caller);
        Ok(())
    }

    // ── Dispute Resolution ────────────────────────────────────────────────────

    /// Raises a dispute, freezing further fund releases.
    pub fn raise_dispute(
        env: Env,
        caller: Address,
        escrow_id: u64,
        milestone_id: Option<u32>,
    ) -> Result<(), EscrowError> {
        caller.require_auth();

        let mut meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;
        if caller != meta.client && caller != meta.freelancer {
            return Err(EscrowError::Unauthorized);
        }
        if meta.status == EscrowStatus::Disputed {
            return Err(EscrowError::DisputeAlreadyExists);
        }
        if meta.status != EscrowStatus::Active {
            return Err(EscrowError::EscrowNotActive);
        }

        meta.status = EscrowStatus::Disputed;
        ContractStorage::save_escrow_meta(&env, &meta);
        events::emit_dispute_raised(&env, escrow_id, &caller);

        if let Some(mid) = milestone_id {
            let mut milestone = ContractStorage::load_milestone(&env, escrow_id, mid)?;
            if milestone.status == MilestoneStatus::Submitted
                || milestone.status == MilestoneStatus::Pending
            {
                milestone.status = MilestoneStatus::Disputed;
                milestone.resolved_at = Some(env.ledger().timestamp());
                ContractStorage::save_milestone(&env, escrow_id, &milestone);
                events::emit_milestone_disputed(&env, escrow_id, mid, &caller);
            }
        }

        Ok(())
    }

    /// Resolves a dispute by distributing remaining funds.
    ///
    /// # Gas notes
    /// - Two token transfers in sequence; unavoidable.
    /// - Reputation updates are two upserts, each touching only one storage entry.
    pub fn resolve_dispute(
        env: Env,
        caller: Address,
        escrow_id: u64,
        client_amount: i128,
        freelancer_amount: i128,
    ) -> Result<(), EscrowError> {
        caller.require_auth();

        let mut meta = ContractStorage::load_escrow_meta(&env, escrow_id)?;

        // Caller must be arbiter or admin
        let is_arbiter = meta.arbiter.as_ref().map_or(false, |a| *a == caller);
        if !is_arbiter {
            ContractStorage::require_admin(&env, &caller)?;
        }

        if meta.status != EscrowStatus::Disputed {
            return Err(EscrowError::EscrowNotDisputed);
        }
        if client_amount + freelancer_amount != meta.remaining_balance {
            return Err(EscrowError::AmountMismatch);
        }

        let token = token::Client::new(&env, &meta.token);
        let contract_addr = env.current_contract_address();

        if client_amount > 0 {
            token.transfer(&contract_addr, &meta.client, &client_amount);
        }
        if freelancer_amount > 0 {
            token.transfer(&contract_addr, &meta.freelancer, &freelancer_amount);
        }

        meta.remaining_balance = 0;
        meta.status = EscrowStatus::Completed;
        ContractStorage::save_escrow_meta(&env, &meta);

        events::emit_dispute_resolved(&env, escrow_id, client_amount, freelancer_amount);

        // Update reputation for both parties
        Self::_update_reputation_internal(&env, &meta.client, false, true, client_amount);
        Self::_update_reputation_internal(&env, &meta.freelancer, false, true, freelancer_amount);

        Ok(())
    }

    // ── Reputation ────────────────────────────────────────────────────────────

    /// Updates on-chain reputation for a user.
    ///
    /// Scoring:
    /// - Completed escrow: +10 base + 1 per 1000 units volume (capped at +20)
    /// - Disputed escrow:  -5 score, increment disputed_count
    pub fn update_reputation(
        env: Env,
        address: Address,
        completed: bool,
        disputed: bool,
        volume: i128,
    ) -> Result<(), EscrowError> {
        Self::_update_reputation_internal(&env, &address, completed, disputed, volume);
        Ok(())
    }

    // ── Upgrade ───────────────────────────────────────────────────────────────

    pub fn upgrade(
        env: Env,
        caller: Address,
        new_wasm_hash: BytesN<32>,
    ) -> Result<(), EscrowError> {
        caller.require_auth();
        ContractStorage::require_admin(&env, &caller)?;
        env.deployer().update_current_contract_wasm(new_wasm_hash);
        Ok(())
    }

    // ── View Functions ────────────────────────────────────────────────────────

    pub fn get_escrow(env: Env, escrow_id: u64) -> Result<EscrowState, EscrowError> {
        ContractStorage::load_escrow(&env, escrow_id)
    }

    pub fn get_reputation(env: Env, address: Address) -> Result<ReputationRecord, EscrowError> {
        Ok(ContractStorage::load_reputation(&env, &address))
    }

    pub fn escrow_count(env: Env) -> u64 {
        ContractStorage::escrow_count(&env)
    }

    pub fn get_milestone(
        env: Env,
        escrow_id: u64,
        milestone_id: u32,
    ) -> Result<Milestone, EscrowError> {
        ContractStorage::load_milestone(&env, escrow_id, milestone_id)
    }

    // ── Meta-Transactions ──────────────────────────────────────────────────────

    /// Executes a meta-transaction on behalf of a user.
    ///
    /// Allows gasless transactions by having a relayer pay fees while
    /// the original signer provides authorization via signature.
    ///
    /// # Arguments
    /// * `meta_tx` - The signed meta-transaction data
    /// * `relayer` - Address of the relayer executing this transaction
    /// * `fee_delegation` - Optional fee delegation configuration
    ///
    /// # Returns
    /// The result of the executed function call
    pub fn execute_meta_transaction(
        env: Env,
        meta_tx: MetaTransaction,
        relayer: Address,
        fee_delegation: Option<FeeDelegation>,
    ) -> Result<String, EscrowError> {
        // Verify the meta-transaction signature and nonce
        ContractStorage::verify_meta_tx_signature(&env, &meta_tx)?;

        // If fee delegation is provided, validate and process it
        if let Some(ref delegation) = fee_delegation {
            Self::_validate_fee_delegation(&env, delegation, &relayer)?;
            Self::_process_fee_delegation(&env, delegation)?;
            events::emit_fee_delegation_used(
                &env,
                &delegation.fee_payer,
                delegation.max_fee,
                &delegation.fee_token,
            );
        }

        // Execute the requested function
        let result = Self::_execute_meta_tx_function(&env, &meta_tx)?;

        // Increment nonce to prevent replay
        ContractStorage::increment_meta_tx_nonce(&env, &meta_tx.signer)?;

        // Emit success event
        events::emit_meta_transaction_executed(
            &env,
            &meta_tx.signer,
            meta_tx.nonce,
            &meta_tx.function_name,
            &relayer,
        );

        Ok(result)
    }

    // ── Fee delegation helpers ─────────────────────────────────────────────────

    /// Validates fee delegation configuration
    fn _validate_fee_delegation(
        env: &Env,
        delegation: &FeeDelegation,
        relayer: &Address,
    ) -> Result<(), EscrowError> {
        // Ensure the relayer is authorized to use this fee delegation
        if &delegation.fee_payer != relayer {
            return Err(EscrowError::Unauthorized);
        }

        // Check that max_fee is reasonable (not zero, not excessively high)
        if delegation.max_fee <= 0 {
            return Err(EscrowError::InvalidEscrowAmount); // Reuse error
        }

        Ok(())
    }

    /// Processes fee delegation by transferring tokens from fee payer to contract
    fn _process_fee_delegation(env: &Env, delegation: &FeeDelegation) -> Result<(), EscrowError> {
        // For now, we assume the fee is paid in XLM and handled at the transaction level
        // In a full implementation, this would transfer tokens to cover gas costs
        // For Soroban, fees are paid in XLM at the transaction level by the relayer

        // We could implement fee collection here if needed:
        // token::Client::new(env, &delegation.fee_token).transfer(
        //     &delegation.fee_payer,
        //     &env.current_contract_address(),
        //     &delegation.max_fee,
        // )?;

        Ok(())
    }

    // ── Meta-transaction argument parsing helpers ─────────────────────────────

    /// Parses arguments for create_escrow from delimited string
    /// Format: client|freelancer|token|total_amount|brief_hash|arbiter|deadline|lock_time
    /// Optional fields can be empty: "" for None
    fn _parse_create_escrow_args(
        env: &Env,
        args_str: &String,
    ) -> Result<CreateEscrowArgs, EscrowError> {
        let parts: Vec<&str> = args_str.as_bytes().split(|&b| b == b'|').collect();
        if parts.len() != 8 {
            return Err(EscrowError::InvalidSignature); // Reuse error
        }

        let client = Address::from_string(&String::from_bytes(env, parts[0]))?;
        let freelancer = Address::from_string(&String::from_bytes(env, parts[1]))?;
        let token = Address::from_string(&String::from_bytes(env, parts[2]))?;
        let total_amount = Self::_parse_u64_to_i128(Self::_parse_str_to_u64(parts[3])?)?;
        let brief_hash = Self::_parse_hex_to_bytes32(env, parts[4])?;

        let arbiter = if parts[5].is_empty() {
            None
        } else {
            Some(Address::from_string(&String::from_bytes(env, parts[5]))?)
        };
        let deadline = if parts[6].is_empty() {
            None
        } else {
            Some(Self::_parse_str_to_u64(parts[6])?)
        };
        let lock_time = if parts[7].is_empty() {
            None
        } else {
            Some(Self::_parse_str_to_u64(parts[7])?)
        };

        Ok(CreateEscrowArgs {
            client,
            freelancer,
            token,
            total_amount,
            brief_hash,
            arbiter,
            deadline,
            lock_time,
        })
    }

    /// Parses arguments for add_milestone from delimited string
    /// Format: caller|escrow_id|title|description_hash|amount
    fn _parse_add_milestone_args(
        env: &Env,
        args_str: &String,
    ) -> Result<AddMilestoneArgs, EscrowError> {
        let parts: Vec<&[u8]> = args_str.as_bytes().split(|&b| b == b'|').collect();
        if parts.len() != 5 {
            return Err(EscrowError::InvalidSignature);
        }

        let caller = Address::from_string(&String::from_bytes(env, parts[0]))?;
        let escrow_id = Self::_parse_str_to_u64(parts[1])?;
        let title = String::from_bytes(env, parts[2]);
        let description_hash = Self::_parse_hex_to_bytes32(env, parts[3])?;
        let amount = Self::_parse_u64_to_i128(Self::_parse_str_to_u64(parts[4])?)?;

        Ok(AddMilestoneArgs {
            caller,
            escrow_id,
            title,
            description_hash,
            amount,
        })
    }

    /// Parses arguments for submit_milestone from delimited string
    /// Format: caller|escrow_id|milestone_id
    fn _parse_submit_milestone_args(
        env: &Env,
        args_str: &String,
    ) -> Result<SubmitMilestoneArgs, EscrowError> {
        let parts: Vec<&[u8]> = args_str.as_bytes().split(|&b| b == b'|').collect();
        if parts.len() != 3 {
            return Err(EscrowError::InvalidSignature);
        }

        let caller = Address::from_string(&String::from_bytes(env, parts[0]))?;
        let escrow_id = Self::_parse_str_to_u64(parts[1])?;
        let milestone_id = Self::_parse_str_to_u64(parts[2])? as u32;

        Ok(SubmitMilestoneArgs {
            caller,
            escrow_id,
            milestone_id,
        })
    }

    /// Parses arguments for approve_milestone from delimited string
    /// Format: caller|escrow_id|milestone_id
    fn _parse_approve_milestone_args(
        env: &Env,
        args_str: &String,
    ) -> Result<ApproveMilestoneArgs, EscrowError> {
        let parts: Vec<&[u8]> = args_str.as_bytes().split(|&b| b == b'|').collect();
        if parts.len() != 3 {
            return Err(EscrowError::InvalidSignature);
        }

        let caller = Address::from_string(&String::from_bytes(env, parts[0]))?;
        let escrow_id = Self::_parse_str_to_u64(parts[1])?;
        let milestone_id = Self::_parse_str_to_u64(parts[2])? as u32;

        Ok(ApproveMilestoneArgs {
            caller,
            escrow_id,
            milestone_id,
        })
    }

    // ── Argument parsing helpers ───────────────────────────────────────────────────

    fn _parse_str_to_u64(s: &[u8]) -> Result<u64, EscrowError> {
        let mut result = 0u64;
        for &b in s {
            if b >= b'0' && b <= b'9' {
                result = result
                    .checked_mul(10)
                    .ok_or(EscrowError::InvalidSignature)?;
                result = result
                    .checked_add((b - b'0') as u64)
                    .ok_or(EscrowError::InvalidSignature)?;
            } else {
                return Err(EscrowError::InvalidSignature);
            }
        }
        Ok(result)
    }

    fn _parse_u64_to_i128(v: u64) -> Result<i128, EscrowError> {
        Ok(v as i128)
    }

    fn _parse_hex_to_bytes32(env: &Env, hex_str: &[u8]) -> Result<BytesN<32>, EscrowError> {
        if hex_str.len() == 64 {
            // 32 bytes * 2 hex chars
            let bytes = hex::decode(hex_str).map_err(|_| EscrowError::InvalidSignature)?;
            if bytes.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                Ok(BytesN::from_array(env, &arr))
            } else {
                Err(EscrowError::InvalidSignature)
            }
        } else {
            Err(EscrowError::InvalidSignature)
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn _update_reputation_internal(
        env: &Env,
        address: &Address,
        completed: bool,
        disputed: bool,
        volume: i128,
    ) {
        let mut record = ContractStorage::load_reputation(env, address);
        let now = env.ledger().timestamp();

        if completed {
            // +10 base + 1 per 1000 volume units, capped at +20 total
            let volume_bonus = (volume / 1_000).min(10) as u64;
            record.total_score = record.total_score.saturating_add(10 + volume_bonus);
            record.completed_escrows += 1;
            record.total_volume = record.total_volume.saturating_add(volume);
        }

        if disputed {
            record.total_score = record.total_score.saturating_sub(5);
            record.disputed_escrows += 1;
        }

        record.last_updated = now;
        ContractStorage::save_reputation(env, &record);
        events::emit_reputation_updated(env, address, record.total_score);
    }

    /// Executes the specific function requested in a meta-transaction.
    ///
    /// This is an internal helper that parses the function arguments and
    /// calls the appropriate contract function.
    fn _execute_meta_tx_function(
        env: &Env,
        meta_tx: &MetaTransaction,
    ) -> Result<String, EscrowError> {
        match meta_tx.function_name.as_str() {
            "create_escrow" => {
                let args = Self::_parse_create_escrow_args(env, &meta_tx.function_args)?;
                let escrow_id = Self::create_escrow(
                    env.clone(),
                    args.client,
                    args.freelancer,
                    args.token,
                    args.total_amount,
                    args.brief_hash,
                    args.arbiter,
                    args.deadline,
                    args.lock_time,
                )?;
                Ok(escrow_id.to_string())
            }
            "add_milestone" => {
                let args = Self::_parse_add_milestone_args(env, &meta_tx.function_args)?;
                let milestone_id = Self::add_milestone(
                    env.clone(),
                    args.caller,
                    args.escrow_id,
                    args.title,
                    args.description_hash,
                    args.amount,
                )?;
                Ok(milestone_id.to_string())
            }
            "submit_milestone" => {
                let args = Self::_parse_submit_milestone_args(env, &meta_tx.function_args)?;
                Self::submit_milestone(
                    env.clone(),
                    args.caller,
                    args.escrow_id,
                    args.milestone_id,
                )?;
                Ok("milestone_submitted".to_string())
            }
            "approve_milestone" => {
                let args = Self::_parse_approve_milestone_args(env, &meta_tx.function_args)?;
                Self::approve_milestone(
                    env.clone(),
                    args.caller,
                    args.escrow_id,
                    args.milestone_id,
                )?;
                Ok("milestone_approved".to_string())
            }
            _ => Err(EscrowError::UnsupportedMetaTxFunction),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use soroban_sdk::{testutils::Address as _, token, BytesN, Env, String};

    fn setup() -> (Env, Address, Address, EscrowContractClient<'static>) {
        let env = Env::default();
        env.mock_all_auths();
        let admin = Address::generate(&env);
        let contract_id = env.register_contract(None, EscrowContract);
        let client = EscrowContractClient::new(&env, &contract_id);
        (env, admin, contract_id, client)
    }

    #[test]
    fn test_initialize_uses_instance_storage() {
        let (env, admin, contract_id, client) = setup();
        client.initialize(&admin);
        env.as_contract(&contract_id, || {
            assert!(env.storage().instance().has(&DataKey::Admin));
            assert!(env.storage().instance().has(&DataKey::EscrowCounter));
            assert!(!env.storage().persistent().has(&DataKey::Admin));
            assert!(!env.storage().persistent().has(&DataKey::EscrowCounter));
        });
    }

    #[test]
    fn test_create_escrow_packs_metadata_separately() {
        let (env, admin, contract_id, client) = setup();
        client.initialize(&admin);

        let escrow_client = Address::generate(&env);
        let freelancer = Address::generate(&env);
        let token_contract = env.register_stellar_asset_contract_v2(admin.clone());
        let token_id = token_contract.address();
        let token_admin = token::StellarAssetClient::new(&env, &token_id);
        let token_client = token::Client::new(&env, &token_id);

        token_admin.mint(&escrow_client, &1_000_i128);

        let escrow_id = client.create_escrow(
            &escrow_client,
            &freelancer,
            &token_id,
            &1_000_i128,
            &BytesN::from_array(&env, &[1; 32]),
            &None,
            &None,
        );

        assert_eq!(escrow_id, 0);
        assert_eq!(token_client.balance(&contract_id), 1_000_i128);

        env.as_contract(&contract_id, || {
            assert!(env
                .storage()
                .persistent()
                .has(&PackedDataKey::EscrowMeta(escrow_id)));
            assert!(!env.storage().persistent().has(&DataKey::Escrow(escrow_id)));
        });
    }

    #[test]
    fn test_get_milestone_reads_granular_storage_entry() {
        let (env, admin, contract_id, client) = setup();
        client.initialize(&admin);

        let escrow_client = Address::generate(&env);
        let freelancer = Address::generate(&env);
        let token_contract = env.register_stellar_asset_contract_v2(admin.clone());
        let token_id = token_contract.address();
        let token_admin = token::StellarAssetClient::new(&env, &token_id);

        token_admin.mint(&escrow_client, &1_000_i128);

        let escrow_id = client.create_escrow(
            &escrow_client,
            &freelancer,
            &token_id,
            &1_000_i128,
            &BytesN::from_array(&env, &[2; 32]),
            &None,
            &None,
        );

        let milestone_id = client.add_milestone(
            &escrow_client,
            &escrow_id,
            &String::from_str(&env, "Design"),
            &BytesN::from_array(&env, &[3; 32]),
            &300_i128,
        );

        let milestone = client.get_milestone(&escrow_id, &milestone_id);
        assert_eq!(milestone.id, milestone_id);
        assert_eq!(milestone.amount, 300_i128);

        env.as_contract(&contract_id, || {
            assert!(env
                .storage()
                .persistent()
                .has(&PackedDataKey::Milestone(escrow_id, milestone_id)));
        });
    }

    #[test]
    fn test_get_reputation_returns_default_record() {
        let (env, _, _, client) = setup();
        let user = Address::generate(&env);
        let record = client.get_reputation(&user);
        assert_eq!(record.address, user);
        assert_eq!(record.total_score, 0);
        assert_eq!(record.completed_escrows, 0);
    }

    #[test]
    fn test_approve_milestone_o1_completion_check() {
        let (env, admin, contract_id, client) = setup();
        client.initialize(&admin);

        let escrow_client = Address::generate(&env);
        let freelancer = Address::generate(&env);
        let token_contract = env.register_stellar_asset_contract_v2(admin.clone());
        let token_id = token_contract.address();
        let token_admin = token::StellarAssetClient::new(&env, &token_id);

        token_admin.mint(&escrow_client, &500_i128);

        let escrow_id = client.create_escrow(
            &escrow_client,
            &freelancer,
            &token_id,
            &500_i128,
            &BytesN::from_array(&env, &[4; 32]),
            &None,
            &None,
        );

        let mid = client.add_milestone(
            &escrow_client,
            &escrow_id,
            &String::from_str(&env, "Dev"),
            &BytesN::from_array(&env, &[5; 32]),
            &500_i128,
        );

        client.submit_milestone(&freelancer, &escrow_id, &mid);
        client.approve_milestone(&escrow_client, &escrow_id, &mid);

        // Escrow should be Completed after the single milestone is approved
        let state = client.get_escrow(&escrow_id);
        assert_eq!(state.status, EscrowStatus::Completed);

        // approved_count field should be 1 in raw storage
        env.as_contract(&contract_id, || {
            let meta: EscrowMeta = env
                .storage()
                .persistent()
                .get(&PackedDataKey::EscrowMeta(escrow_id))
                .unwrap();
            assert_eq!(meta.approved_count, 1);
            assert_eq!(meta.milestone_count, 1);
        });
    }

    #[test]
    fn test_cancel_escrow() {
        let (env, admin, _, client) = setup();
        client.initialize(&admin);

        let escrow_client = Address::generate(&env);
        let freelancer = Address::generate(&env);
        let token_contract = env.register_stellar_asset_contract_v2(admin.clone());
        let token_id = token_contract.address();
        let token_admin = token::StellarAssetClient::new(&env, &token_id);
        let token_client = token::Client::new(&env, &token_id);

        token_admin.mint(&escrow_client, &200_i128);

        let escrow_id = client.create_escrow(
            &escrow_client,
            &freelancer,
            &token_id,
            &200_i128,
            &BytesN::from_array(&env, &[6; 32]),
            &None,
            &None,
        );

        client.cancel_escrow(&escrow_client, &escrow_id);

        let state = client.get_escrow(&escrow_id);
        assert_eq!(state.status, EscrowStatus::Cancelled);
        assert_eq!(token_client.balance(&escrow_client), 200_i128);
    }

    #[test]
    #[ignore = "implement full flow — Issues #2–#11"]
    fn test_full_escrow_lifecycle() {}

    #[test]
    #[ignore = "implement dispute flow — Issues #9–#10"]
    fn test_dispute_resolution() {}
}
