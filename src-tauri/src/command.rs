use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Startup {
    Hidden,
    Show,
    Toggle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedCommand {
    Run(Startup),
    Version,
    Help,
    Invalid(String),
}

/// What an already-running shell should do with a forwarded launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Forwarded {
    Show,
    Nothing,
    Toggle,
}

impl fmt::Display for ParsedCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "{message}"),
            _ => write!(formatter, "{self:?}"),
        }
    }
}

pub fn parse<I>(arguments: I) -> ParsedCommand
where
    I: IntoIterator<Item = String>,
{
    let mut arguments = arguments.into_iter();
    let Some(first) = arguments.next() else {
        return ParsedCommand::Run(Startup::Hidden);
    };
    if let Some(extra) = arguments.next() {
        return ParsedCommand::Invalid(format!("unexpected argument: {extra}"));
    }
    match first.as_str() {
        "--hidden" => ParsedCommand::Run(Startup::Hidden),
        "--show" => ParsedCommand::Run(Startup::Show),
        "--toggle" => ParsedCommand::Run(Startup::Toggle),
        "--version" | "-V" => ParsedCommand::Version,
        "--help" | "-h" => ParsedCommand::Help,
        other => ParsedCommand::Invalid(format!("unknown argument: {other}")),
    }
}

/// A launch forwarded by the single-instance plugin. Parsing version, help,
/// and invalid arguments happens before Tauri starts, so they never reach
/// this table.
pub fn forward<I>(arguments: I) -> Forwarded
where
    I: IntoIterator<Item = String>,
{
    let arguments: Vec<String> = arguments.into_iter().collect();
    match arguments.as_slice() {
        [] => Forwarded::Show,
        [argument] if argument == "--show" => Forwarded::Show,
        [argument] if argument == "--hidden" => Forwarded::Nothing,
        [argument] if argument == "--toggle" => Forwarded::Toggle,
        _ => Forwarded::Nothing,
    }
}

pub fn usage() -> &'static str {
    "Usage: verbix [--hidden | --show | --toggle | --version | --help]\n\
     \n\
     --hidden   start in the tray without opening the panel (default)\n\
     --show     open the panel after start\n\
     --toggle   toggle listening after the voice service is ready\n\
     --version  print the version and exit\n\
     --help     print this help and exit\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_starts_hidden() {
        assert_eq!(
            parse(Vec::<String>::new()),
            ParsedCommand::Run(Startup::Hidden)
        );
    }

    #[test]
    fn unknown_argument_is_invalid() {
        assert!(matches!(
            parse(["--wat".to_string()]),
            ParsedCommand::Invalid(_)
        ));
    }
}
