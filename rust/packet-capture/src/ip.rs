use pnet::packet::ethernet::EthernetPacket;
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::ipv4::Ipv4Packet;
use pnet::packet::ipv6::Ipv6Packet;
use pnet::packet::tcp::TcpPacket;
use pnet::packet::udp::UdpPacket;
use pnet::packet::Packet;

const WIDTH: usize = 20;

use super::packets::GettableEndPoints;

/// IPv4パケットを構築し次のレイヤのハンドラを呼び出す
pub fn ipv4_handler(ethernet: &EthernetPacket) {
    if let Some(packet) = Ipv4Packet::new(ethernet.payload()) {
        match packet.get_next_level_protocol() {
            IpNextHeaderProtocols::Tcp => {
                tcp_handler(&packet);
            }
            IpNextHeaderProtocols::Udp => {
                udp_handler(&packet);
            }
            _ => {
                info!("Not a TCP or UDP packet");
            }
        }
    }
}

/// IPv6パケットを構築し次のレイヤのハンドラを呼び出す
pub fn ipv6_handler(ethernet: &EthernetPacket) {
    if let Some(packet) = Ipv6Packet::new(ethernet.payload()) {
        match packet.get_next_header() {
            IpNextHeaderProtocols::Tcp => {
                tcp_handler(&packet);
            }
            IpNextHeaderProtocols::Udp => {
                udp_handler(&packet);
            }
            _ => {
                info!("Not a TCP or UDP packet");
            }
        }
    }
}

/// TCPパケットを構築する
fn tcp_handler(packet: &dyn GettableEndPoints) {
    let tcp = TcpPacket::new(packet.get_payload());
    if let Some(tcp) = tcp {
        print_packet_info(packet, &tcp, "TCP");
    }
}

/// UDPパケットを構築する
fn udp_handler(packet: &dyn GettableEndPoints) {
    let udp = UdpPacket::new(packet.get_payload());
    if let Some(udp) = udp {
        print_packet_info(packet, &udp, "UDP");
    }
}

/// アプリケーション層のデータをバイナリで表示する
fn print_packet_info(l3: &dyn GettableEndPoints, l4: &dyn GettableEndPoints, proto: &str) {
    println!(
        "Captured a {} packet from {}|{} to {}|{}",
        proto,
        l3.get_source(),
        l4.get_source(),
        l3.get_destination(),
        l4.get_destination()
    );

    let payload = l4.get_payload();
    let len = payload.len();
    for i in 0..len {
        print!("{:<02X} ", payload[i]);
        if i % WIDTH == WIDTH - 1 || i == len - 1 {
            for _ in 0..WIDTH - 1 - i % WIDTH {
                print!("   ");
            }
            print!("| ");
            for j in i - i % WIDTH..=i {
                if payload[j].is_ascii_alphabetic() {
                    print!("{}", payload[j] as char);
                } else {
                    print!(".");
                }
            }
            println!();
        }
    }
    println!("{}", "=".repeat(WIDTH * 3));
    println!();
}
