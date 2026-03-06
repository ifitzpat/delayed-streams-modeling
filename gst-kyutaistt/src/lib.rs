// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use gst::glib;
use gst::prelude::*;

mod imp;
mod model;
mod vad;

glib::wrapper! {
    pub struct KyutaiStt(ObjectSubclass<imp::KyutaiStt>) @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "kyutaistt",
        gst::Rank::NONE,
        KyutaiStt::static_type(),
    )?;
    Ok(())
}

gst::plugin_define!(
    kyutaistt,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
