//
//  RouteInfoView.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit

final class RouteInfoView: UIView {
    private enum Layout {
        static let textLeftInset: CGFloat = 30
        static let textRightInset: CGFloat = 0
        static let titleBottomInset: CGFloat = 8
        static let sectionHeight: CGFloat = 36

        static let yourLocationImageSize = CGSize(width: 11, height: 11)
        static let sourceImageSize = CGSize(width: 6, height: 6)
        static let stopsImageSize = CGSize(width: 19, height: 19)
        static let destinationImageSize = CGSize(width: 16, height: 16)

        static let imageCenterX: CGFloat = 9.5
    }

    private let yourLocationLabel = UILabel()
    private let walkLabel = UILabel()
    private let sourceLabel = UILabel()
    private let busLabel = UILabel()
    private let stopsLabel = UILabel()
    private let destionationLabel = UILabel()

    private let yourLocationImageView = UIImageView()
    private let sourceImageView = UIImageView()
    private let stopsImageView = UIImageView()
    private let destionationImageView = UIImageView()

    private let yourLocationToSourceLineView = DottedLineView()
    private let sourceToStopsLineView = UIView()
    private let stopsToDestinationLineView = UIView()

    init() {
        super.init(frame: .zero)

        self.setupSubviews()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()

        self.yourLocationImageView.frame = CGRect(origin: .zero, size: Layout.yourLocationImageSize)
        self.yourLocationImageView.frame.centerX = Layout.imageCenterX
        self.yourLocationImageView.frame.centerY = self.yourLocationLabel.frame.center.y

        self.sourceImageView.frame = CGRect(origin: .zero, size: Layout.sourceImageSize)
        self.sourceImageView.frame.centerX = Layout.imageCenterX
        self.sourceImageView.frame.centerY = self.sourceLabel.center.y

        self.stopsImageView.frame = CGRect(origin: .zero, size: Layout.stopsImageSize)
        self.stopsImageView.frame.centerX = Layout.imageCenterX
        self.stopsImageView.frame.centerY = self.stopsLabel.center.y

        self.destionationImageView.frame = CGRect(origin: .zero, size: Layout.destinationImageSize)
        self.destionationImageView.frame.centerX = Layout.imageCenterX
        self.destionationImageView.frame.centerY = self.destionationLabel.center.y

        self.yourLocationToSourceLineView.frame = CGRect(
            origin: .zero,
            size: CGSize(width: 1, height: self.sourceImageView.frame.minY - self.yourLocationImageView.frame.maxY + 2)
        )
        self.yourLocationToSourceLineView.frame.centerX = Layout.imageCenterX
        self.yourLocationToSourceLineView.frame.origin.y = self.yourLocationImageView.frame.maxY - 1

        self.sourceToStopsLineView.frame = CGRect(
            origin: .zero,
            size: CGSize(width: 1, height: self.stopsImageView.frame.minY - self.sourceImageView.frame.maxY - 4)
        )
        self.sourceToStopsLineView.frame.centerX = Layout.imageCenterX
        self.sourceToStopsLineView.frame.origin.y = self.sourceImageView.frame.maxY + 2

        self.stopsToDestinationLineView.frame = CGRect(
            origin: .zero,
            size: CGSize(width: 1, height: self.destionationImageView.frame.minY - self.stopsImageView.frame.maxY - 4)
        )
        self.stopsToDestinationLineView.frame.centerX = Layout.imageCenterX
        self.stopsToDestinationLineView.frame.origin.y = self.stopsImageView.frame.maxY + 2
    }

    private func setupSubviews() {
        self.addSubview(self.yourLocationLabel)
        self.addSubview(self.walkLabel)
        self.addSubview(self.sourceLabel)
        self.addSubview(self.busLabel)
        self.addSubview(self.stopsLabel)
        self.addSubview(self.destionationLabel)

        self.addSubview(self.yourLocationImageView)
        self.addSubview(self.sourceImageView)
        self.addSubview(self.stopsImageView)
        self.addSubview(self.destionationImageView)

        self.addSubview(self.yourLocationToSourceLineView)
        self.addSubview(self.sourceToStopsLineView)
        self.addSubview(self.stopsToDestinationLineView)

        self.yourLocationLabel.snp.makeConstraints { make in
            make.top.equalToSuperview()
            make.left.equalToSuperview().offset(Layout.textLeftInset)
            make.right.equalToSuperview().offset(-Layout.textRightInset)
        }

        self.walkLabel.snp.makeConstraints { make in
            make.top.equalTo(self.yourLocationLabel.snp.bottom).offset(Layout.titleBottomInset)
            make.left.equalToSuperview().offset(Layout.textLeftInset)
            make.right.equalToSuperview().offset(-Layout.textRightInset)
        }

        self.sourceLabel.snp.makeConstraints { make in
            make.top.equalTo(self.yourLocationLabel.snp.bottom).offset(Layout.sectionHeight)
            make.left.equalToSuperview().offset(Layout.textLeftInset)
            make.right.equalToSuperview().offset(-Layout.textRightInset)
        }

        self.busLabel.snp.makeConstraints { make in
            make.top.equalTo(self.sourceLabel.snp.bottom).offset(Layout.titleBottomInset)
            make.left.equalToSuperview().offset(Layout.textLeftInset)
            make.right.equalToSuperview().offset(-Layout.textRightInset)
        }

        self.stopsLabel.snp.makeConstraints { make in
            make.top.equalTo(self.sourceLabel.snp.bottom).offset(Layout.sectionHeight)
            make.left.equalToSuperview().offset(Layout.textLeftInset)
            make.right.equalToSuperview().offset(-Layout.textRightInset)
        }

        self.destionationLabel.snp.makeConstraints { make in
            make.top.equalTo(self.stopsLabel.snp.bottom).offset(Layout.sectionHeight)
            make.left.equalToSuperview().offset(Layout.textLeftInset)
            make.right.equalToSuperview().offset(-Layout.textRightInset)
        }

        self.setupLabelsText()
        self.setupImages()
        self.setupLines()
    }

    private func setupLabelsText() {
        self.yourLocationLabel.attributedText = Self.makeTitleAttributedText(value: "Your location", lineHeight: 19)
        self.walkLabel.attributedText = Self.makeDescriptionAttributedText(value: "Walk for 250m, about 2 mins")
        self.sourceLabel.attributedText = Self.makeTitleAttributedText(value: "Aalto University", lineHeight: 19)
        self.busLabel.attributedText = Self.makeDescriptionAttributedText(value: "Bus 550 Elfvik Quarry")
        self.stopsLabel.attributedText = Self.makeTitleAttributedText(value: "4 stops (12 mins)", lineHeight: 19)
        self.destionationLabel.attributedText = Self.makeTitleAttributedText(value: "Elfvik Quarry", lineHeight: 18)

        [
            self.yourLocationLabel, self.walkLabel, self.sourceLabel,
            self.busLabel, self.stopsLabel, self.destionationLabel
        ].forEach { label in
            label.numberOfLines = 0
        }
    }

    private func setupImages() {
        self.yourLocationImageView.image = UIImage(named: "filledCircle")
        self.sourceImageView.image = UIImage(named: "ellipse")
        self.stopsImageView.image = UIImage(named: "busStopIcon")
        self.destionationImageView.image = UIImage(named: "marker")
    }

    private func setupLines() {
        self.yourLocationToSourceLineView.lineColor = .app
        self.yourLocationToSourceLineView.horizontal = false
        self.yourLocationToSourceLineView.round = true
        self.sourceToStopsLineView.backgroundColor = .app
        self.stopsToDestinationLineView.backgroundColor = .app
    }

    private static func makeTitleAttributedText(value: String, lineHeight: CGFloat) -> NSAttributedString {
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.minimumLineHeight = lineHeight
        paragraphStyle.maximumLineHeight = lineHeight

        let attributedText = NSAttributedString(
            string: value,
            attributes: [
                .font: UIFont.appRegular(16),
                .paragraphStyle: paragraphStyle,
                .foregroundColor: UIColor.app
            ]
        )

        return attributedText
    }

    private static func makeDescriptionAttributedText(value: String) -> NSAttributedString {
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.minimumLineHeight = 11
        paragraphStyle.maximumLineHeight = 11

        let attributedText = NSAttributedString(
            string: value,
            attributes: [
                .font: UIFont.appRegular(10),
                .paragraphStyle: paragraphStyle,
                .foregroundColor: UIColor.blackText
            ]
        )

        return attributedText
    }
}
