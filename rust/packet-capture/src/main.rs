use pnet::datalink;
use pnet::datalink::Channel::Ethernet;
use pnet::packet::ethernet::EtherTypes;
use pnet::packet::ethernet::EthernetPacket;
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::ipv4::Ipv4Packet;
use pnet::packet::ipv6::Ipv6Packet;
use std::env;

mod ip;
use ip::ipv4_handler;
use ip::ipv6_handler;

mod packets;

#[macro_use]
extern crate log;
extern crate pnet;

fn main() {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        error!("Usage: {} [interface_name]", args[0]);
        std::process::exit(1);
    }

    let interface_name: &str = &args[1];

    let interfaces = datalink::interfaces();
    let interface = interfaces
        .into_iter()
        .find(|iface| iface.name == *interface_name)
        .expect("Failed to get interface");

    // データリンクのチャネルを取得
    let (_tx, mut rx) = match datalink::channel(&interface, Default::default()) {
        Ok(Ethernet(tx, rx)) => (tx, rx),
        Ok(_) => panic!("Unhandled channel"),
        Err(e) => panic!("Failed to create datalink channel {}", e),
    };

    loop {
        match rx.next() {
            Ok(frame) => {
                // 受信データからイーサネットフレームの構築
                let frame = EthernetPacket::new(frame).unwrap();
                match frame.get_ethertype() {
                    EtherTypes::Ipv4 => {
                        ipv4_handler(&frame);
                    }
                    EtherTypes::Ipv6 => {
                        ipv6_handler(&frame);
                    }
                    _ => {
                        info!("Not an IPv4 or IPv6 packet");
                    }
                }
            }
            Err(e) => {
                error!("Failed to read: {}", e);
            }
        }
    }
}
