//! BytePlus REST client: direct Seedance/Seedream access (optional, cheaper
//! than fal if confirmed in M0).

/// Optional; implement ImageProvider/VideoProvider if M0 picks direct access.
pub struct BytePlusClient {
    pub(crate) http: reqwest::Client,
    pub(crate) key: String,
}
