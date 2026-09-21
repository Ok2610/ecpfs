use super::*;

use std::cell::Cell;

/// `store_err_with` does not call its closure when the result is `Ok`.
#[test]
fn store_err_with_does_not_build_its_context_on_ok() {
    let calls = Cell::new(0);

    let result = Ok::<u32, String>(7).store_err_with(|| {
        calls.set(calls.get() + 1);
        "context".to_string()
    });

    assert_eq!(result.unwrap(), 7);
    assert_eq!(calls.get(), 0, "context was built although nothing failed");
}

/// `store_err` and `store_err_with` return the same `EcpError::Store` message.
#[test]
fn store_err_and_store_err_with_format_the_same_message() {
    let eager = Err::<u32, _>("disk full").store_err("failed to write /a");
    let lazy = Err::<u32, _>("disk full").store_err_with(|| "failed to write /a".to_string());

    for result in [eager, lazy] {
        match result.unwrap_err() {
            EcpError::Store(message) => assert_eq!(message, "failed to write /a: disk full"),
            other => panic!("expected Store, got {other:?}"),
        }
    }
}
